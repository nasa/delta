"""
Take an input folder of image files and
convert them to a Tensorflow friendly format in the output folder.
"""
import os
import sys
import argparse
import zipfile
import multiprocessing
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# TODO: Make sure this goes everywhere!
if sys.version_info < (3, 0, 0):
    print('\nERROR: Must use Python version >= 3.0.')
    sys.exit(1)

from delta.imagery import utilities #pylint: disable=C0413
from delta.imagery.sources import landsat #pylint: disable=C0413
from delta.imagery.sources import worldview #pylint: disable=C0413
from delta.imagery.sources import landsat_toa #pylint: disable=C0413
from delta.imagery.sources import worldview_toa #pylint: disable=C0413
import tfrecord_convert_image #pylint: disable=C0413


#------------------------------------------------------------------------------


def convert_file_tif(input_path, output_path, work_folder, tile_size): #pylint: disable=W0613
    """Convert one input tif image"""
    print(input_path)
    # This is for simple images so the only thing to do is TFRecord conversion
    tfrecord_convert_image.tiff_to_tf_record([input_path], output_path, tile_size)

    print('Finished converting file ', input_path)
    return 0


#TODO: Can move some of this out of /bin, but need to move the TOA stuff first.
def convert_file_landsat(input_path, output_path, work_folder, tile_size):
    """Convert one input Landsat file (containing multiple tif tiles)"""

    # Create the folders
    if not os.path.exists(work_folder):
        os.mkdir(work_folder)

    # Unzip the input file
    print('Untar file: ', input_path)
    utilities.unpack_to_folder(input_path, work_folder)

    meta_path  = landsat.find_mtl_file(work_folder)
    meta_data  = landsat.parse_mtl_file(meta_path)
    scene_info = landsat.get_scene_info(input_path)
    bands_to_use = landsat.get_landsat_bands_to_use(scene_info['sensor'])

    # Apply TOA conversion
    print('TOA conversion...')
    toa_folder = os.path.join(work_folder, 'toa_output')
    landsat_toa.do_landsat_toa_conversion(meta_path, toa_folder, calc_reflectance=True, num_processes=1)

    if not landsat.check_if_files_present(meta_data, toa_folder):
        raise Exception('TOA conversion failed for: ', input_path)
    print('TOA conversion finished')

    toa_paths = landsat.get_band_paths(meta_data, toa_folder, bands_to_use)

    # Convert the image into a multi-part binary TFRecord file that can be easily read in by TensorFlow.
    tfrecord_convert_image.tiff_to_tf_record(toa_paths, output_path, tile_size)

    # Remove all of the temporary files
    os.system('rm -rf ' + work_folder)
    print('Finished converting file ', input_path)
    return 0


#TODO: Can move some of this out of /bin, but need to move the TOA stuff first.
def convert_file_worldview(input_path, output_path, work_folder, tile_size):
    """Convert one input WorldView file"""

    # Create the folders
    if not os.path.exists(work_folder):
        os.mkdir(work_folder)

    toa_path = os.path.join(work_folder, 'toa.tif')
    scene_info = worldview.get_scene_info(input_path)

    # Unzip the input file
    print('Unzip file: ', input_path)
    with zipfile.ZipFile(input_path, 'r') as zip_ref:
        zip_ref.extractall(work_folder)

    (tif_path, meta_path) = worldview.get_files_from_unpack_folder(work_folder)

    # Apply TOA conversion

    # TODO: Any benefit to passing in the tile size here?
    print('TOA conversion...')
    # TODO get reflectance working!
    worldview_toa.do_worldview_toa_conversion(tif_path, meta_path, toa_path, calc_reflectance=False)
    if not os.path.exists(toa_path):
        raise Exception('TOA conversion failed for: ', input_path)
    print('TOA conversion finished')

    # Convert the image into a multi-part binary TFRecord file that can be easily read in by TensorFlow.
    bands_to_use = worldview.get_worldview_bands_to_use(scene_info['sensor'])
    tfrecord_convert_image.tiff_to_tf_record([toa_path], output_path, tile_size, bands_to_use)

    # Remove all of the temporary files
    os.system('rm -rf ' + work_folder)
    print('Finished converting file ', input_path)
    return 0


# Cleaner ways to do this don't work with multiprocessing!
def try_catch_and_call(func, input_path, output_path, work_folder, tile_size):
    """Wrap the provided function in a try/catch statement"""
    try:
        return func(input_path, output_path, work_folder, tile_size)
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
                            help="Specify the input image type [worldview, landsat, tif].")

        #parser.add_argument("--num-threads", dest="num_threads", type=int, default=1,
        #                    help="Number of threads to use per process.")

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

    INPUT_EXTENSIONS  = {'worldview':'.zip', 'landsat':'.gz', 'tif':'.tif'}
    CONVERT_FUNCTIONS = {'worldview':convert_file_worldview,
                         'landsat':convert_file_landsat,
                         'tif':convert_file_tif}

    if options.image_type not in INPUT_EXTENSIONS:
        print('Unrecognized image type: ' + options.image_type)
        return -1


    convert_file_function = CONVERT_FUNCTIONS[options.image_type]

    # Find all of the input files to process with full paths
    input_extension = INPUT_EXTENSIONS[options.image_type]
    input_files = utilities.get_files_with_extension(options.input_folder, input_extension)
    print('Found ', len(input_files), ' input files to convert')

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
                                                                     output_path, work_folder, options.tile_size)))
        else:
            result = try_catch_and_call(convert_file_function, f, output_path, work_folder, options.tile_size)
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
