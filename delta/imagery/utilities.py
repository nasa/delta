"""
Miscellaneous utility classes/functions.
"""
import os
import sys
import shutil
import zipfile
import tarfile

def unpack_to_folder(compressed_path, unpack_folder):
    """
    Unpack a file into the given folder.
    """

    tmpdir = os.path.normpath(unpack_folder) + '_working'

    ext = os.path.splitext(compressed_path)[1]
    try:
        if ext.lower() == '.zip':
            with zipfile.ZipFile(compressed_path, 'r') as zf:
                zf.extractall(tmpdir)
        else: # Assume a tar file
            with tarfile.TarFile(compressed_path, 'r') as tf:
                tf.extractall(tmpdir)
    except:
        shutil.rmtree(tmpdir)
        raise
    # make this atomic so we don't have incomplete data
    os.rename(tmpdir, unpack_folder)

def progress_bar(text, fill_amount, prefix = '', length = 80): #pylint: disable=W0613
    """
    Prints a progress bar. Call multiple times with increasing progress to
    overwrite the printed line.
    """
    filled_length = int(length * fill_amount)
    fill_char = '█' if sys.stdout.encoding.lower() == 'utf-8' else 'X'
    prog_bar = fill_char * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s' % (prefix, prog_bar, text), end = '\r')
