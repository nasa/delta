"""
Miscellaneous utility classes/functions.
"""
import os
import time
import math
import signal
from osgeo import gdal
import psutil
import numpy as np

from delta.imagery.rectangle import Rectangle

#============================================================================
# Constants

BYTES_PER_MB = 1024*1024
BYTES_PER_GB = 1024*1024*1024

#============================================================================
# Functions

def get_num_bytes_from_gdal_type(gdal_type):
    """Return the number of bytes for one pixel (one band) in a GDAL type."""
    results = {
        gdal.GDT_Byte:    1,
        gdal.GDT_UInt16:  2,
        gdal.GDT_UInt32:  4,
        gdal.GDT_Float32: 4,
        gdal.GDT_Float64: 8
    }
    return results.get(gdal_type)

def get_gdal_data_type(type_str):

    s = type_str.lower()
    if s in ['byte', 'uint8']:
        return gdal.GDT_Byte
    if s in ['short', 'uint16']:
        return gdal.GDT_UInt16
    if s == 'uint32':
        return gdal.GDT_UInt32
    if s in ['float', 'float32']:
        return gdal.GDT_Float32
    if s == 'float64':
        return gdal.GDT_Float64
    raise Exception('Unrecognized data type string: ' + type_str)

def numpy_dtype_to_gdal_type(dtype):

    if dtype == np.uint8:
        return gdal.GDT_Byte
    if dtype == np.uint16:
        return gdal.GDT_UInt16
    if dtype == np.uint32:
        return gdal.GDT_UInt32
    if dtype == np.float32:
        return gdal.GDT_Float32
    if dtype == np.float64:
        return gdal.GDT_Float64
    raise Exception('Unrecognized numpy data type: ' + str(dtype))

def gdal_dtype_to_numpy_type(dtype):

    if dtype == gdal.GDT_Byte:
        return np.uint8
    if dtype == gdal.GDT_UInt16:
        return np.uint16
    if dtype == gdal.GDT_UInt32:
        return np.uint32
    if dtype == gdal.GDT_Float32:
        return np.float32
    if dtype == gdal.GDT_Float64:
        return np.float64
    raise Exception('Unrecognized gdal data type: ' + str(dtype))


def get_pbs_node_list():
    """Get the list of machines we have access to in a PBS job"""

    # When running a PBS job, the list of machines is contained in $PBS_NODEFILE
    #  but there can be duplicate entries in the file.
    node_list = []
    list_path = os.environ['PBS_NODEFILE']
    with os.open(list_path, 'r') as f:
        for line in f:
            entry = line.strip()
            if entry not in node_list:
                node_list.append(entry)
    return node_list


#======================================================
# Functions copied from ASP

def file_is_good(path):
    '''Make sure file exists and is non-empty.'''
    return os.path.exists(path) and (os.path.getsize(path) > 0)

def logger_print(logger, msg):
    '''Print to logger, if present. This helps keeps all messages in sync.'''
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

# This block of code is just to get a non-blocking keyboard check!
class AlarmException(Exception):
    pass
def alarmHandler(signum, frame):
    raise AlarmException
def nonBlockingRawInput(prompt='', timeout=20):
    '''Return a key if pressed or an empty string otherwise.
       Waits for timeout, non-blocking.'''
    signal.signal(signal.SIGALRM, alarmHandler)
    signal.alarm(timeout)
    try:
        text = input(prompt)
        signal.alarm(0)
        return text
    except AlarmException:
        pass # Timeout
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    return ''

def waitForTaskCompletionOrKeypress(taskHandles, logger = None, interactive=True,
                                    quitKey='q', sleepTime=5):
    '''Block in this function until the user presses a key or all tasks complete'''

    # Wait for all the tasks to complete
    notReady = len(taskHandles)
    while notReady > 0:

        if interactive:
            # Wait and see if the user presses a key
            msg = ('Waiting on ' + str(notReady) + ' process(es), press '
                   +str(quitKey)+'<Enter> to abort...\n')
            keypress = nonBlockingRawInput(prompt=msg, timeout=sleepTime)
            if keypress == quitKey:
                logger_print(logger, 'Recieved quit command!')
                break
        else:
            logger_print(logger, "Waiting on " + str(notReady) + ' incomplete tasks.')
            time.sleep(sleepTime)

        # As long as we have this process waiting, keep track of our resource consumption.
        cpuPercentUsage = psutil.cpu_percent()
        memInfo         = psutil.virtual_memory()
        memUsed         = memInfo[0] - memInfo[1]
        memPercentUsage = float(memUsed) / float(memInfo[0])

        usageMessage = ('CPU percent usage = %f, Memory percent usage = %f'
                        % (cpuPercentUsage, memPercentUsage))
        logger_print(logger, usageMessage)

        # Otherwise count up the tasks we are still waiting on.
        notReady = 0
        for task in taskHandles:
            if not task.ready():
                notReady += 1

def stop_task_pool(pool):
    """Stop remaining tasks and kill the pool"""

    PROCESS_POOL_KILL_TIMEOUT = 3
    pool.close()
    time.sleep(PROCESS_POOL_KILL_TIMEOUT)
    pool.terminate()
    pool.join()


def unpack_to_folder(compressed_path, unpack_folder):
    """Unpack a file into the given folder"""

    # Force create the output folder
    os.system('mkdir -p ' + unpack_folder)

    ext = os.path.splitext(compressed_path)[1]
    if ext.lower() == '.zip':
        cmd = 'unzip ' + compressed_path + ' -d ' + unpack_folder
    else: # Assume a tar file
        cmd = 'tar -xf ' + compressed_path + ' --directory ' + unpack_folder

    print(cmd)
    os.system(cmd)

def get_files_with_extension(folder, extension):
    """Return paths of all the files in a directory matching an extension (recursive)"""

    input_files = []
    for root, dummy_directories, filenames in os.walk(folder):
        for filename in filenames:
            if os.path.splitext(filename)[1] == extension:
                path = os.path.join(root, filename)
                input_files.append(path)
    return input_files

#======================================================================================
# Functions for working with image chunks.

def generate_chunk_info(chunk_size, chunk_overlap):
    """From chunk size and overlap,
       compute chunk_start_offset and chunk_spacing.
    """
    chunk_start_offset = int(math.floor(chunk_size / 2))
    chunk_spacing      = chunk_size - chunk_overlap
    return (chunk_start_offset, chunk_spacing)


def get_chunk_center_list_in_region(region_rect, chunk_start_offset,
                                    chunk_spacing, chunk_size):
    """Return a list of (x,y) chunk centers, one for each chunk that is centered
       in the region Rectangle.  This function does not prune out chunks that
       extend out past the right and bottom boundaries of the image.
    """

    # Figure out the first x/y center that falls in the region.
    init_x = (int(math.ceil((region_rect.min_x - chunk_start_offset) / chunk_spacing))
              * chunk_spacing + chunk_start_offset)
    init_y = (int(math.ceil((region_rect.min_y - chunk_start_offset) / chunk_spacing))
              * chunk_spacing + chunk_start_offset)

    # Also get the last x/y center in the region
    last_x = (int(math.floor((region_rect.max_x-1 - init_x) / chunk_spacing))
              * chunk_spacing + init_x)
    last_y = (int(math.floor((region_rect.max_y-1 - init_y) / chunk_spacing))
              * chunk_spacing + init_y)

    # Find the bounding box that includes the full chunks from this region.
    chunk_bbox = Rectangle(init_x, init_y, init_x, init_y)
    for x in [init_x, last_x]:
        for y in [init_y, last_y]:
            rect = rect_from_chunk_center((x,y), chunk_size)
            chunk_bbox.expand_to_contain_rect(rect)

    # Generate a list of all of the centers in the region.
    center_list = []
    y = init_y
    while y < region_rect.max_y:
        x = init_x
        while x < region_rect.max_x:
            if not region_rect.contains_pt(x, y):
                raise Exception('Contain error: %d, %d, %s' % (x, y, str(region_rect)))
            center_list.append((x,y))
            x += chunk_spacing
        y += chunk_spacing

    return (center_list, chunk_bbox)

def rect_from_chunk_center(center, chunk_size):
    """Given a chunk center and size, get the bounding Rectangle"""

    # Offset of the center coord from the top left coord.
    chunk_center_offset = int(math.floor(chunk_size / 2))

    (x, y) = center
    min_x = x+chunk_center_offset-chunk_size
    min_y = y+chunk_center_offset-chunk_size
    return Rectangle(min_x, min_y, min_x+chunk_size, min_y+chunk_size)

def restrict_chunk_list_to_roi(chunk_center_list, chunk_size, roi):
    """Remove all chunks from the list which extend out past the ROI"""
    output_list = []
    output_chunk_roi = None
    for center in chunk_center_list:
        rect = rect_from_chunk_center(center, chunk_size)
        if roi.contains_rect(rect):
            output_list.append(center)
            if output_chunk_roi is None:
                output_chunk_roi = rect
            else:
                output_chunk_roi.expand_to_contain_rect(rect)
    return (output_list, output_chunk_roi)
