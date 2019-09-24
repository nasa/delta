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
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from delta.imagery import utilities #pylint: disable=C0413
from delta.imagery import tfrecord_conversions #pylint: disable=C0413


#------------------------------------------------------------------------------


def compress_and_delete(input_path, output_path, keep):
    """Convert the TFRecord file and then delete the input file"""
    result = tfrecord_conversions.compress_tfrecord_file(input_path, output_path)
    if (result > 0) and not keep: #Always keep on failure
        os.remove(input_path)
    return result

def parallel_compress_tfrecords(input_paths, output_paths, num_processes, redo=False, keep=False):
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
            task_handles.append(pool.apply_async(compress_and_delete, (input_path, output_path, keep)))
        else:
            result = compress_and_delete(input_path, output_path, keep)
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



def get_input_files(options):
    """Return the list of input files from the specified source"""

    if (not options.input_folder) and (not options.input_file_list):
        print('ERROR: must provide either --input-folder or --input-file-list')
        return []

    if options.input_file_list:
        input_files = []
        with open(options.input_file_list, 'r') as f:
            for line in f:
                input_files.append(line.strip())

    else: # Use the input folder
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
                return []

        # Find all of the input files to process with full paths
        input_files = utilities.get_files_with_extension(options.input_folder, input_extension)

    return input_files


# Cleaner ways to do this don't work with multiprocessing!
def try_catch_and_call(func, input_path, output_paths, work_folder):
    """Wrap the provided function in a try/catch statement"""
    try:
        func(input_path, output_paths, work_folder)
        print('Finished converting file ', input_path)
        return 0
    except Exception:  #pylint: disable=W0703
        print('ERROR: Failed to process input file: ' + input_path)
        traceback.print_exc()
        sys.stdout.flush()
        return -1

def main(argsIn): #pylint: disable=R0914,R0912

    try:
        usage  = "usage: convert_input_image_folder.py [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--input-folder", dest="input_folder", required=True,
                            help="Path to the folder containing compressed images, output files will."
                            + " be written in the relative arrangement to this folder.")

        parser.add_argument("--input-file-list", dest="input_file_list", default=None,
                            help="Path to file listing all of the compressed image paths.")

        parser.add_argument("--output-folder", dest="output_folder", required=True,
                            help="Where to write the converted output images.")

        parser.add_argument("--mix-outputs", action="store_true", dest="mix_outputs", default=False,
                            help="Instead of copying input folder structure, mix up the output tiles in one folder.")

        parser.add_argument("--compress-only", action="store_true", dest="compress_only", default=False,
                            help="Skip straight to compressing uncompressed TFRecord files.")

        parser.add_argument("--keep", action="store_true", dest="keep", default=False,
                            help="Don't delete the uncompressed TFRecord files.")

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

        parser.add_argument("--limit", dest="limit", type=int, default=None,
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

    input_files = get_input_files(options) # DEBUG
    print('Found ', len(input_files), ' input files to convert')
    if not input_files:
        return -1

    # Prepopulate some conversion function arguments
    convert_file_function = \
      functools.partial(tfrecord_conversions.convert_image_to_tfrecord,
                        tile_size=options.tile_size, image_type=options.image_type)

    mix_paths = []
    output_prefix = os.path.join(options.output_folder, 'mix_file_uncompressed_')
    for i in range(0, len(input_files)):
        this_path = ('%s%08d%s' % (output_prefix, i, output_extension))
        mix_paths.append(this_path)

    if not options.compress_only:

        # Set up processing pool
        if options.num_processes > 1:
            print('Starting processing pool with ' + str(options.num_processes) +' processes.')
            pool = multiprocessing.Pool(options.num_processes)
            task_handles = []

        # Assign input files to the pool
        count = 0
        num_succeeded = 0
        for f in input_files:

            if options.mix_outputs: # Use flat folder with mixed files
                output_paths = mix_paths
                work_folder =  tempfile.mkdtemp()
            else:
                # Recreate the subfolders in the output folder
                rel_path    = os.path.relpath(f, options.input_folder)
                output_path = os.path.join(options.output_folder, rel_path)
                output_path = os.path.splitext(output_path)[0] + output_extension
                subfolder   = os.path.dirname(output_path)
                name        = os.path.basename(output_path)
                name        = os.path.splitext(name)[0]
                work_folder = os.path.join(options.output_folder, name + '_work')
                output_paths = output_path

                if not os.path.exists(subfolder):
                    os.mkdir(subfolder)

                # Skip existing output files
                if os.path.exists(output_path) and not options.redo:
                    continue

            if options.num_processes > 1:# Add the command to the task pool
                task_handles.append(pool.apply_async(try_catch_and_call,(convert_file_function, f,
                                                                         output_paths, work_folder)))
            else:
                result = try_catch_and_call(convert_file_function, f, output_paths, work_folder)
                #result = convert_file_function(f, output_paths, work_folder)
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

    if not options.mix_outputs:
        return 0 # Finished!

    print('Now need to compress the mix files...')

    compressed_paths = []
    for p in mix_paths:
        new_path = p.replace('uncompressed_', '')
        compressed_paths.append(new_path)

    parallel_compress_tfrecords(mix_paths, compressed_paths,
                                options.num_processes, options.redo, options.keep)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
