#!/usr/bin/python
"""
Take an input folder of image files and
convert them to a Tensorflow friendly format in the output folder.
"""
import os
import sys
import argparse
import multiprocessing
import signal
import traceback
import functools

from delta.config import config
from delta.imagery import utilities
from delta.imagery import tfrecord_conversions

#------------------------------------------------------------------------------


def compress_and_delete(input_path, output_path):
    """Convert the TFRecord file and then delete the input file"""
    result = tfrecord_conversions.compress_tfrecord_file(input_path, output_path)
    if result > 0: #Always keep on failure
        os.remove(input_path)
    return result

def parallel_compress_tfrecords(input_paths, output_paths, num_processes):
    """Use multiple processes to compress a list of TFrecord files"""

    # Set up processing pool
    if num_processes > 1:
        print('Starting processing pool with ' + str(num_processes) +' processes.')
        pool = multiprocessing.Pool(num_processes)
        task_handles = []

    # Assign input files to the pool
    count = 0
    num_succeeded = 0
    for input_path, output_path in zip(input_paths, output_paths):

        if num_processes > 1: # Add the command to the task pool
            task_handles.append(pool.apply_async(compress_and_delete, (input_path, output_path)))
        else:
            result = compress_and_delete(input_path, output_path)
            if result > 0:
                num_succeeded += 1

        count += 1

    if num_processes > 1:
        # Wait for all the tasks to complete
        print('Finished adding ' + str(len(task_handles)) + ' tasks to the pool.')
        utilities.waitForTaskCompletionOrKeypress(task_handles, interactive=False)

        # All tasks should be finished, clean up the processing pool
        utilities.stop_task_pool(pool)

        num_succeeded = 0
        for h in task_handles:
            if h.get() > 0:
                num_succeeded += 1

    print('Successfully compressed ', num_succeeded, ' out of ', count, ' input files.')


# Cleaner ways to do this don't work with multiprocessing!
def try_catch_and_call(func, input_path, output_paths):
    """Wrap the provided function in a try/catch statement"""
    try:
        func(input_path, output_paths)
        print('Finished converting file ', input_path)
        return 0
    except: # pylint:disable=bare-except
        print('ERROR: Failed to process input file: ' + input_path)
        traceback.print_exc()
        # delete any incomplete work so we don't use it
        try:
            for filename in output_paths:
                os.remove(filename)
        except OSError:
            pass
        sys.stdout.flush()
        return -1

def convert_images(input_files, output_paths, image_type, tile_size, num_processes, mix_outputs):
    convert_image_function = functools.partial(tfrecord_conversions.convert_image_to_tfrecord,
                                               tile_size=tile_size, image_type=image_type)
    def init_worker():
        signal.signal(signal.SIGTERM, lambda x, y: os.kill(os.getpid(), signal.SIGINT))

    # Set up processing pool
    pool = multiprocessing.Pool(num_processes, init_worker)

    try:
        results = pool.starmap_async(convert_image_function, zip(input_files, output_paths))
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

    parallel_compress_tfrecords(input_files[0], map(input_files[0], lambda x : x.replace('uncompressed_', '')),
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

def write_config_file(output_folder, image_type):
    with open(os.path.join(output_folder, 'dataset.cfg'), 'w') as f:
        f.write('[input_dataset]\n\n')
        f.write('extension=.tfrecord\n')
        f.write('file_type=tfrecord\n')
        f.write('image_type=%s\n' % (image_type))
        f.write('data_directory=.\n\n')
        f.write('label_extension=_label.tfrecord\n')
        f.write('label_type=tfrecord\n')
        f.write('label_directory=.\n')

def main(argsIn): #pylint: disable=R0914,R0912

    usage  = "usage: convert_input_image_folder.py [options]"
    parser = argparse.ArgumentParser(usage=usage)

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

    parser.add_argument("--limit", dest="limit", type=int, default=None,
                        help="Only try to convert this many files before stopping.")

    options = config.parse_args(parser, argsIn, ml=False)

    # Make sure the output folder exists
    if not os.path.exists(options.output_folder):
        os.mkdir(options.output_folder)

    inputs = config.dataset()
    (image_files, label_files) = inputs.images()
    if label_files is None:
        label_files = []
    else:
        assert len(label_files) == len(image_files)

    if len(label_files) > 0 and options.mix_outputs:
        raise NotImplementedError('Cannot support labels with mix_outputs.')

    image_files = list(image_files[:options.limit])
    label_files = list(label_files[:options.limit])

    output_paths = tfrecord_paths(options.mix_outputs, image_files, inputs.data_directory(),
                                  options.output_folder, '.tfrecord')
    output_label_paths = list(map(lambda x : list(map(lambda y: y.replace('.tfrecord', '_label.tfrecord'), x)),
                                  output_paths))
    if not options.redo and not options.mix_outputs:
        for i in range(len(image_files) - 1, -1, -1):
            if os.path.exists(output_paths[i][0]):
                print('Skipping %s which already exists. Use --redo to overwrite.' % (output_paths[i][0]))
                output_paths.pop(i)
                image_files.pop(i)
        for i in range(len(label_files) - 1, -1, -1):
            if os.path.exists(output_label_paths[i][0]):
                print('Skipping %s which already exists. Use --redo to overwrite.' % (output_label_paths[i][0]))
                output_label_paths.pop(i)
                label_files.pop(i)

    print('Converting %s images and %s labels...' % (len(image_files), len(label_files)))
    if len(image_files) == 0:
        return -1

    convert_images(image_files, output_paths, inputs.file_type(), options.tile_size, options.num_processes,
                   options.mix_outputs)
    if len(label_files) > 0:
        convert_images(label_files, output_label_paths, inputs.label_type(),
                       options.tile_size, options.num_processes, options.mix_outputs)

    write_config_file(options.output_folder, inputs.file_type())

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
