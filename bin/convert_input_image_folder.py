#!/usr/bin/python
"""
Take an input folder of image files and
convert them to a Tensorflow friendly format in the output folder.
"""
import os
import sys
import argparse
import multiprocessing
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

def parallel_compress_tfrecords(input_paths, output_paths, num_processes, redo=False):
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

        # Skip existing output files
        if os.path.exists(output_path) and not redo:
            continue

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

def convert_images(input_files, base_folder, output_folder, output_extension, image_type, tile_size, \
                   num_processes, mix_outputs, limit, redo):#pylint:disable=too-many-arguments,too-many-locals,too-many-branches
    convert_image_function = \
      functools.partial(tfrecord_conversions.convert_image_to_tfrecord,
                        tile_size=tile_size, image_type=image_type)

    mix_paths = []
    output_prefix = os.path.join(output_folder, 'mix_file_uncompressed_')
    for i in range(0, len(input_files)):
        this_path = ('%s%08d%s' % (output_prefix, i, output_extension))
        mix_paths.append(this_path)

    # Set up processing pool
    pool = multiprocessing.Pool(num_processes)
    task_handles = []

    # Assign input files to the pool
    count = 0
    num_succeeded = 0
    for f in input_files:

        if mix_outputs: # Use flat folder with mixed files
            output_paths = mix_paths
        else:
            if base_folder:
                # Recreate the subfolders in the output folder
                rel_path    = os.path.relpath(f, base_folder)
                output_path = os.path.join(output_folder, rel_path)
                output_path = os.path.splitext(output_path)[0] + output_extension
                subfolder   = os.path.dirname(output_path)

                if not os.path.exists(subfolder):
                    os.mkdir(subfolder)
            else:
                output_path = os.path.basename(f)
                output_path = os.path.join(output_folder, output_path)

            # Skip existing output files
            if os.path.exists(output_path) and not redo:
                print('Skipping %s, since %s already exists. Use --redo to overwrite.' % (f, output_path))
                continue
            output_paths = [output_path]

        task_handles.append(pool.apply_async(try_catch_and_call,(convert_image_function, f,
                                                                 output_paths)))

        count += 1
        if limit and (count >= limit):
            print('Limiting processing to ', limit, ' files.')
            break

    # Wait for all the tasks to complete
    utilities.waitForTaskCompletionOrKeypress(task_handles, interactive=False)

    # All tasks should be finished, clean up the processing pool
    utilities.stop_task_pool(pool)

    num_succeeded = 0
    for h in task_handles:
        if h.get() == 0:
            num_succeeded += 1

    if num_succeeded != count:
        print('Failed to convert %s / %s files.' % (count - num_succeeded, count), file=sys.stderr)
        return 1
    print('Successfully converted %s files.' % (num_succeeded))

    if not mix_outputs:
        return 0 # Finished!

    print('Compressing the mixed files...')

    compressed_paths = []
    for p in mix_paths:
        new_path = p.replace('uncompressed_', '')
        compressed_paths.append(new_path)

    parallel_compress_tfrecords(mix_paths, compressed_paths,
                                num_processes, redo)

    return 0

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
    print('Converting %s images and %s labels...' % (len(image_files), len(label_files) if label_files else 0))
    if len(image_files) == 0:
        return -1

    convert_images(image_files, inputs.data_directory(), options.output_folder, '.tfrecord',
                   inputs.file_type(), options.tile_size, options.num_processes,
                   options.mix_outputs, options.limit, options.redo)
    if label_files:
        convert_images(label_files, inputs.label_directory(), options.output_folder, '.tfrecordlabel',
                       inputs.label_file_type(), options.tile_size, options.num_processes,
                       options.mix_outputs, options.limit, options.redo)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
