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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# TODO: Make sure this goes everywhere!
if sys.version_info < (3, 0, 0):
    print('\nERROR: Must use Python version >= 3.0.')
    sys.exit(1)

from delta.imagery import utilities #pylint: disable=C0413
from delta.imagery import tfrecord_conversions #pylint: disable=C0413


#------------------------------------------------------------------------------

# Cleaner ways to do this don't work with multiprocessing!
def try_catch_and_call(func, input_path, output_path, work_folder):
    """Wrap the provided function in a try/catch statement"""
    try:
        func(input_path, output_path, work_folder)
        print('Finished converting file ', input_path)
        return 0
    except Exception:  #pylint: disable=W0703
        print('ERROR: Failed to process input file: ' + input_path)
        traceback.print_exc()
        sys.stdout.flush()
        return -1

def main(argsIn):

    try:

        usage  = "usage: convert_input_image_folder.py [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--input-folder", dest="input_folder", required=True,
                            help="Path to the folder containing compressed images.")

        parser.add_argument("--output-folder", dest="output_folder", required=True,
                            help="Where to write the converted output images.")

        parser.add_argument("--image-type", dest="image_type", required=True,
                            help="Specify the input image type [worldview, landsat, tif, rgba].")

        parser.add_argument("--extension", dest="input_extension", default=None,
                            help="Manually specify the input extension instead of using the default.")

        parser.add_argument("--tile-size", nargs=2, metavar=('tile_width', 'tile_height'),
                            dest='tile_size', default=[256, 256], type=int,
                            help="Specify the size of the tiles the input images will be split up into.")

        parser.add_argument("--redo", action="store_true", dest="redo", default=False,
                            help="Re-write already existing output files.")

        parser.add_argument("--labels", action="store_true", dest="labels", default=False,
                            help="Set this when the input files are label files.")

        parser.add_argument("--num-processes", dest="num_processes", type=int, default=1,
                            help="Number of parallel processes to use.")

        parser.add_argument("--limit", dest="limit", type=int, default=1,
                            help="Only try to convert this many files before stopping.")

        options = parser.parse_args(argsIn)

    except argparse.ArgumentError:
        print(usage)
        return -1

    # Make sure the output folder exists
    if not os.path.exists(options.output_folder):
        os.mkdir(options.output_folder)

    output_extension = '.tfrecord'
    if options.labels:
        output_extension = '.tfrecordlabel'

    # Figure out the input extension to use
    DEFAULT_EXTENSIONS = {'worldview':'.zip', 'landsat':'.gz', 'tif':'.tif', 'rgba':'.tif'}
    if options.input_extension:
        input_extension = options.input_extension
    else:
        try:
            input_extension = DEFAULT_EXTENSIONS[options.image_type]
            print('Using the default input extension: ', input_extension)
        except KeyError:
            print('Unrecognized image type: ' + options.image_type)
            return -1

    # Find all of the input files to process with full paths
    input_files = utilities.get_files_with_extension(options.input_folder, input_extension)
    print('Found ', len(input_files), ' input files to convert')

    # Prepopulate some conversion function arguments
    convert_file_function = \
      functools.partial(tfrecord_conversions.convert_image_to_tfrecord,
                        tile_size=options.tile_size, image_type=options.image_type)

    # Set up processing pool
    if options.num_processes > 1:
        print('Starting processing pool with ' + str(options.num_processes) +' processes.')
        pool = multiprocessing.Pool(options.num_processes)
        task_handles = []

    # Assign input files to the pool
    count = 0
    num_succeeded = 0
    for f in input_files:

        # Recreate the subfolders in the output folder
        rel_path    = os.path.relpath(f, options.input_folder)
        output_path = os.path.join(options.output_folder, rel_path)
        output_path = os.path.splitext(output_path)[0] + output_extension
        subfolder   = os.path.dirname(output_path)
        name        = os.path.basename(output_path)
        name        = os.path.splitext(name)[0]
        work_folder = os.path.join(options.output_folder, name + '_work')

        if not os.path.exists(subfolder):
            os.mkdir(subfolder)

        # Skip existing output files
        if os.path.exists(output_path) and not options.redo:
            continue

        if options.num_processes > 1:# Add the command to the task pool
            task_handles.append(pool.apply_async(try_catch_and_call,(convert_file_function, f,
                                                                     output_path, work_folder)))
        else:
            result = try_catch_and_call(convert_file_function, f, output_path, work_folder)
            if result == 0:
                num_succeeded += 1
        #break # DEBUG

        count += 1
        if options.limit and (count >= options.limit):
            print('Limiting processing to ', options.limit, ' files.')
            break

    if options.num_processes > 1:
        # Wait for all the tasks to complete
        print('Finished adding ' + str(len(task_handles)) + ' tasks to the pool.')
        utilities.waitForTaskCompletionOrKeypress(task_handles, interactive=False)

        # All tasks should be finished, clean up the processing pool
        utilities.stop_task_pool(pool)

        num_succeeded = 0
        for h in task_handles:
            if h.get() == 0:
                num_succeeded += 1

    print('Successfully converted ', num_succeeded, ' out of ', count, ' input files.')
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
