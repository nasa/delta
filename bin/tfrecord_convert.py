#!/usr/bin/python3
"""
Take a set of images and convert them to a Tensorflow friendly format.
"""
import os
import sys
import argparse
import multiprocessing
import signal
import functools

import tensorflow as tf

from delta.config import config
from delta.imagery.sources import loader, tfrecord

def compress_and_delete(input_path, output_path):
    """Convert the TFRecord file and then delete the input file"""
    writer   = tfrecord.make_tfrecord_writer(output_path, compress=True)
    reader   = tf.data.TFRecordDataset(input_path, compression_type="")
    iterator = reader.make_one_shot_iterator()

    next_element = iterator.get_next()
    sess = tf.Session()

    count = 0
    while True:
        try:
            value = sess.run(next_element)
            writer.write(value)
            count += 1
        except tf.errors.OutOfRangeError:
            break
    if count > 0: #Always keep on failure
        os.remove(input_path)
    return count

def __init_worker():
    signal.signal(signal.SIGTERM, lambda x, y: os.kill(os.getpid(), signal.SIGINT))

def parallel_compress_tfrecords(input_paths, output_paths, num_processes):
    """Use multiple processes to compress a list of TFrecord files"""

    pool = multiprocessing.Pool(num_processes, __init_worker)
    try:
        results = pool.starmap_async(compress_and_delete, zip(input_paths, output_paths))
        results.get()
    except:
        pool.terminate()
        pool.join()
        raise

    pool.close()
    pool.join()
    if not results.successful():
        print('Compression failed.')
        sys.exit(0)

    print('Successfully compressed input files.')

def convert_image(ds_config, is_label, output_paths, tile_size, mix_outputs, redo, image_id):
    if not redo and not mix_outputs:
        if os.path.exists(output_paths[image_id][0]):
            print('Skipping %s which already exists. Use --redo to overwrite.' % (output_paths[image_id][0]))
            return (3, 4)
    if is_label:
        image = loader.load_label(ds_config, image_id)
    else:
        image = loader.load_image(ds_config, image_id)
    try:
        tfrecord.image_to_tfrecord(image, output_paths[image_id], tile_size, show_progress=True)
    except:
        for f in output_paths:
            os.remove(f)
        raise
    return (1, 2)

def convert_images(ds_config, is_label, output_paths, tile_size, num_processes, mix_outputs, redo):

    # Set up processing pool
    pool = multiprocessing.Pool(num_processes, __init_worker)

    convert_function = functools.partial(convert_image, ds_config, is_label, output_paths, tile_size, mix_outputs, redo)
    try:
        results = pool.map_async(convert_function, range(len(output_paths)))
        results.get()
    except:
        pool.terminate()
        pool.join()
        raise

    pool.close()
    pool.join()
    if not results.successful():
        print('Conversion failed.')
        sys.exit(0)

    print('Successfully converted all files.')

    if not mix_outputs:
        return 0 # Finished!

    print('Compressing the mixed files...')

    parallel_compress_tfrecords(output_paths, map(output_paths, lambda x : x.replace('uncompressed_', '')),
                                num_processes)

    return 0

def to_output_path(p, base_folder, output_folder, output_extension):
    if base_folder:
        # Recreate the subfolders in the output folder
        rel_path    = os.path.relpath(p, base_folder)
        output_path = os.path.join(output_folder, rel_path)
        output_path = os.path.splitext(output_path)[0] + output_extension
        subfolder   = os.path.dirname(output_path)

        if not os.path.exists(subfolder):
            os.mkdir(subfolder)
    else:
        output_path = os.path.basename(p)
        output_path = os.path.join(output_folder, output_path)
    return [output_path]

def to_mix_paths(output_folder, output_extension, num_files):
    mix_paths = []
    output_prefix = os.path.join(output_folder, 'mix_file_uncompressed_')
    for i in range(0, num_files):
        this_path = ('%s%08d%s' % (output_prefix, i, output_extension))
        mix_paths.append(this_path)
    return mix_paths

def tfrecord_paths(mix_files, files, base_folder, output_folder, output_extension):
    if mix_files:
        # return a list of n lists to pass in each time
        return [to_mix_paths(output_folder, output_extension, len(files))] * len(files)
    return list(map(functools.partial(to_output_path, base_folder=base_folder,
                                      output_folder=output_folder, output_extension=output_extension),
                    files))

def write_config_file(output_folder):
    with open(os.path.join(output_folder, 'dataset.cfg'), 'w') as f:
        f.write('[input_dataset]\n\n')
        f.write('extension=.tfrecord\n')
        f.write('image_type=tfrecord\n')
        f.write('data_directory=.\n\n')
        f.write('label_extension=_label.tfrecord\n')
        f.write('label_type=tfrecord\n')
        f.write('label_directory=.\n')

def main(args):
    parser = argparse.ArgumentParser(description='Convert images to tfrecord files.')

    parser.add_argument("--output-folder", dest="output_folder", required=True,
                        help="Where to write the converted output images.")

    parser.add_argument("--mix-outputs", action="store_true", dest="mix_outputs", default=False,
                        help="Instead of copying input folder structure, mix up the output tiles in one folder.")

    parser.add_argument("--tile-size", nargs=2, metavar=('tile_width', 'tile_height'),
                        dest='tile_size', default=[256, 256], type=int,
                        help="Specify the size of the tiles the input images will be split up into.")

    parser.add_argument("--redo", action="store_true", dest="redo", default=False,
                        help="Re-write already existing output files.")

    parser.add_argument("--num-processes", dest="num_processes", type=int, default=1,
                        help="Number of parallel processes to use.")

    options = config.parse_args(parser, args, ml=False)

    # Make sure the output folder exists
    if not os.path.exists(options.output_folder):
        os.mkdir(options.output_folder)

    inputs = config.dataset()
    (image_files, label_files) = inputs.images()

    if label_files is not None and options.mix_outputs:
        raise NotImplementedError('Cannot support labels with mix_outputs.')

    if label_files is None:
        label_files = []
    else:
        assert len(label_files) == len(image_files)

    output_paths = tfrecord_paths(options.mix_outputs, image_files, inputs.data_directory(),
                                  options.output_folder, '.tfrecord')
    output_label_paths = list(map(lambda x : list(map(lambda y: y.replace('.tfrecord', '_label.tfrecord'), x)),
                                  output_paths))

    print('Converting %s images and %s labels...' % (len(image_files), len(label_files)))

    convert_images(inputs, False, output_paths, options.tile_size, options.num_processes,
                   options.mix_outputs, options.redo)
    if label_files:
        convert_images(inputs, True, output_label_paths, options.tile_size, options.num_processes,
                       options.mix_outputs, options.redo)

    write_config_file(options.output_folder)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
