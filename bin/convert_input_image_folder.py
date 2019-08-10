"""
Take an input folder of image files and
convert them to a Tensorflow friendly format in the output folder.
"""
import os
import sys
import argparse
import zipfile
import multiprocessing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# TODO: Make sure this goes everywhere!
if sys.version_info < (3, 0, 0):
    print('\nERROR: Must use Python version >= 3.0.')
    sys.exit(1)

from delta.imagery import utilities #pylint: disable=C0413
import landsat_toa #pylint: disable=C0413
import worldview_toa #pylint: disable=C0413
import tfrecord_convert_image #pylint: disable=C0413


#------------------------------------------------------------------------------


def convert_file_tif(input_path, output_path, work_folder, tile_size): #pylint: disable=W0613
    """Convert one input tif image"""
    print(input_path)
    # This is for simple images so the only thing to do is TFRecord conversion
    tfrecord_convert_image.tiff_to_tf_record([input_path], output_path, tile_size)

    print('Finished converting file ', input_path)


def convert_file_landsat(input_path, output_path, work_folder, tile_size):
    """Convert one input Landsat file (containing multiple tif tiles)"""

    # Create the folders
    if not os.path.exists(work_folder):
        os.mkdir(work_folder)

    # Unzip the input file
    print('Untar file: ', input_path)
    utilities.unpack_to_folder(input_path, work_folder)

    file_list = os.listdir(work_folder)
    meta_files = [f for f in file_list if 'MTL.txt' in f]
    if len(meta_files) != 1:
        raise Exception('Error processing ', input_path, ', meta list: ', str(file_list))
    meta_path = os.path.join(work_folder, meta_files[0])

    # Apply TOA conversion
    print('TOA conversion...')
    toa_folder = os.path.join(work_folder, 'toa_output')
    landsat_toa.do_work(meta_path, toa_folder, calc_reflectance=True)
    if not os.path.exists(toa_folder):
        raise Exception('TOA conversion failed for: ', input_path)
    print('TOA conversion finished')

    # Get the TOA output files (one per band)
    file_list = os.listdir(toa_folder)
    toa_paths = [os.path.join(toa_folder, f) for f in file_list if '.TIF' in f]
    # Convert the image into a multi-part binary TFRecord file that can be easily read in by TensorFlow.
    tfrecord_convert_image.tiff_to_tf_record(toa_paths, output_path, tile_size)

    # Remove all of the temporary files
    os.system('rm -rf ' + work_folder)
    print('Finished converting file ', input_path)


def convert_file_worldview(input_path, output_path, work_folder, tile_size):
    """Convert one input WorldView file"""

    # Create the folders
    if not os.path.exists(work_folder):
        os.mkdir(work_folder)

    toa_path = os.path.join(work_folder, 'toa.tif')

    # Unzip the input file
    print('Unzip file: ', input_path)
    with zipfile.ZipFile(input_path, 'r') as zip_ref:
        zip_ref.extractall(work_folder)

    file_list = os.listdir(work_folder)
    tif_files = [f for f in file_list if (os.path.splitext(f)[1] == '.tif' and f != 'toa.tif')]
    if len(tif_files) != 1:
        raise Exception('Error processing ', input_path, ', file list: ', str(file_list))

    vendor_folder = os.path.join(work_folder, 'vendor_metadata')
    file_list = os.listdir(vendor_folder)
    meta_files = [f for f in file_list if os.path.splitext(f)[1] == '.IMD']
    if len(meta_files) != 1:
        raise Exception('Error processing ', input_path, ', meta list: ', str(file_list))

    tif_path  = os.path.join(work_folder,   tif_files[0])
    meta_path = os.path.join(vendor_folder, meta_files[0])

    # Apply TOA conversion

    # TODO: Any benefit to passing in the tile size here?
    print('TOA conversion...')
    worldview_toa.do_work(tif_path, meta_path, toa_path, calc_reflectance=False) # TODO get reflectance working!
    if not os.path.exists(toa_path):
        raise Exception('TOA conversion failed for: ', input_path)
    print('TOA conversion finished')

    # Convert the image into a multi-part binary TFRecord file that can be easily read in by TensorFlow.
    tfrecord_convert_image.tiff_to_tf_record([toa_path], output_path, tile_size)

    # Remove all of the temporary files
    os.system('rm -rf ' + work_folder)
    print('Finished converting file ', input_path)

def main(argsIn):

    try:

        usage  = "usage: convert_input_image_folder.py [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--input-folder", dest="input_folder", required=True,
                            help="Path to the folder containing compressed images.")

        parser.add_argument("--output-folder", dest="output_folder", required=True,
                            help="Where to write the converted output images.")

        parser.add_argument("--num-processes", dest="num_processes", type=int, default=1,
                            help="Number of parallel processes to use.")

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

    # Get the list of zip files in the input folder
    input_extension = INPUT_EXTENSIONS[options.image_type]
    input_files = []
    for root, dummy_directories, filenames in os.walk(options.input_folder):
        for filename in filenames:
            if os.path.splitext(filename)[1] == input_extension:
                path = os.path.join(root, filename)
                input_files.append(path)
    print('Found ', len(input_files), ' input files to convert')

    # Set up processing pool
    print('Starting processing pool with ' + str(options.num_processes) +' processes.')
    pool = multiprocessing.Pool(options.num_processes)
    task_handles = []

    # Assign input files to the pool
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

        # Add the command to the task pool
        task_handles.append(pool.apply_async(convert_file_function, (f, output_path, work_folder, options.tile_size)))
        #convert_file_function(f, output_path, work_folder, options.tile_size) # DEBUG
        #break # DEBUG

    # Wait for all the tasks to complete
    print('Finished adding ' + str(len(task_handles)) + ' tasks to the pool.')
    utilities.waitForTaskCompletionOrKeypress(task_handles, interactive=False)

    # All tasks should be finished, clean up the processing pool
    utilities.stop_task_pool(pool)


    print('Folder conversion is finished.')
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
