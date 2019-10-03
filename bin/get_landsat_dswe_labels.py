#!/usr/bin/python
"""
Script to fetch DSWE (label) images from USGS corresponding to a given Landsat image.
"""
import os
import sys
import argparse
import subprocess

import gdal
from osgeo import osr

from usgs import api

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# TODO: Why is the path not being set correctly???
os.environ['PATH'] = os.environ['PATH'].replace('/nobackup/smcmich1/code/anaconda3\\Library\\bin;','').replace('\\','/')
#print('PATH = ' + str(os.environ['PATH']))

from delta.imagery import utilities  #pylint: disable=C0413
from delta.imagery.sources import landsat #pylint: disable=C0413

#------------------------------------------------------------------------------

def look_for_file(folder, contains):
    """Return the name of a file inside folder that has all strings
       in the 'contains' list.
    """

    files = os.listdir(folder)
    for f in files:
        good = True
        for c in contains:
            if c not in f:
                good = False
                break
        if good:
            return os.path.join(folder, f)
    return None


def unpack_inputs(tar_folder, unpack_folder):
    """Make sure all of the input label files are untarred.
       The unpack folder can be the same as the tar folder.
       Returns the list of label files.
    """

    if not os.path.exists(unpack_folder):
        os.mkdir(unpack_folder)

    file_list = []

    # Loop through tar files
    input_list = os.listdir(tar_folder   )

    for f in input_list:
        ext = os.path.splitext(f)[1]
        if ext != '.tar':
            continue
        # The name of the input tar does not fully match the untar file names
        name   = os.path.basename(f)
        parts  = name.split('_')
        prefix = '_'.join(parts[0:4])

        # Look to see if we have a matching label file
        tar_path   = os.path.join(tar_folder, f)
        label_path = look_for_file(unpack_folder, [prefix, '_INWM.tif'])

        # If we did not find the INWM file, untar.
        if not label_path:
            utilities.unpack_to_folder(tar_path, unpack_folder)
            # Look again for a matching INWM file
            label_path = look_for_file(unpack_folder, [prefix, '_INWM.tif'])
            if not label_path:
                raise Exception('Failed to untar label file: ' + tar_path)
        file_list.append(label_path)

    return file_list


def get_bounding_coordinates(landsat_path, convert_to_lonlat):
    """Return the lower left and upper right lonlat coordinates of the landsat file"""

    # Get the projected coordinates
    src = gdal.Open(landsat_path)
    ulx, xres, dummyxskew, uly, dummyyskew, yres  = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)

    if convert_to_lonlat:
        source = osr.SpatialReference()
        source.ImportFromWkt(src.GetProjection())

        target = osr.SpatialReference()
        target.ImportFromEPSG(4326) # The standard lonlat projection

        transform = osr.CoordinateTransformation(source, target)

        # Transform the point. You can also create an ogr geometry and use the more generic `point.Transform()`
        (lrx, lry, dummy_h) = transform.TransformPoint(lrx, lry)
        (ulx, uly, dummy_h) = transform.TransformPoint(ulx, uly)

    return ((ulx, lry), (lrx, uly)) # Switch the corners


def fetch_dswe_images(date, ll_coord, ur_coord, output_folder, user, password, force_login):
    """Download all DSWE images that fit the given criteria to the output folder
       if they are not already present.  The coordinates must be in lon/lat degrees.
    """

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    CATALOG = 'EE'
    DATASET = 'SP_TILE_DSWE'

    # Only log in if our session expired (ugly function use to check!)
    if force_login or (not api._get_api_key(None)): #pylint: disable=W0212
        print('Logging in to USGS EarthExplorer...')
        dummy_result = api.login(user, password, save=True, catalogId=CATALOG) #pylint: disable=W0612

        #print(api._get_api_key(None))
        #raise Exception('DEBUG')

    print('Submitting EarthExplorer query...')
    results = api.search(DATASET, CATALOG, where={}, start_date=date, end_date=date,
                         ll=dict([('longitude',ll_coord[0]),('latitude',ll_coord[1])]),
                         ur=dict([('longitude',ur_coord[0]),('latitude',ur_coord[1])]),
                         max_results=12, extended=False)

    if not results['data']:
        raise Exception('Did not find any DSWE data that matched the Landsat file!')
    print('Found ' + str(len(results['data']['results'])) + ' matching files.')

    for scene in results['data']['results']:
        #print('------------')
        #print(scene)
        print('Found match: ' + scene['entityId'])

        fname = scene['entityId'] + '.tar'
        output_path = os.path.join(output_folder, fname)

        if os.path.exists(output_path):
            print('Already have image on disk!')
            continue

        r = api.download(DATASET, CATALOG, [scene['entityId']], product='DSWE')
        print(r)
        if not r['data']:
            raise Exception('Failed to get download URL!')
        url = r['data'][0]['url']
        cmd = ('wget "%s" --user %s --password %s -O %s' % (url, user, password, output_path))
        print(cmd)
        os.system(cmd)

        if not os.path.exists(output_path):
            raise Exception('Failed to download file ' + output_path)

    print('Finished downloading DSWE files.')
    # Can just let this time out
    #api.logout()



def main(argsIn):

    try:

        usage  = "usage: get_landsat_dswe_labels [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--output-path", dest="output_path", required=True,
                            help="Output image path.")

        parser.add_argument("--landsat-path", dest="landsat_path", required=True,
                            help="Path to the landsat image we want the label to match.")

        parser.add_argument("--label-folder", dest="label_folder", required=True,
                            help="Download DSWE files to this folder.")

        parser.add_argument("--user", dest="user", required=False,
                            help="User name for EarthExplorer website, needed to download new files.")
        parser.add_argument("--password", dest="password", required=False,
                            help="Password name for EarthExplorer website, needed to download new files.")

        parser.add_argument("--force-login", action="store_true",
                            dest="force_login", default=False,
                            help="Don't reuse the cached EE API key if present.")

        #parser.add_argument("--download-files", action="store_true",
        #                    dest="download_files", default=False,
        #                    help="Download new DSWE files if they are not already there.")


        options = parser.parse_args(argsIn)

    except argparse.ArgumentError:
        print(usage)
        return -1

    output_folder = os.path.dirname(options.output_path)
    if output_folder and not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Extract information about the landsat file
    date = landsat.get_scene_info(options.landsat_path)['date']
    date = date[0:4] + '-' + date[4:6] + '-' + date[6:8]#  '2018-12-26'
    (ll_coord, ur_coord) = get_bounding_coordinates(options.landsat_path,
                                                    convert_to_lonlat=True)

    if options.user and options.password:
        print('Login info provided, searching for overlapping label images...')
        fetch_dswe_images(date, ll_coord, ur_coord, options.label_folder,
                          options.user, options.password, options.force_login)
    else:
        print('--user and --password not provided, skipping label download step.')

    # Untar the input files if needed
    untar_folder = options.label_folder
    input_files = unpack_inputs(options.label_folder, untar_folder)
    if not input_files:
        print('Did not detect any input label files!')
        return -1

    merge_path = options.output_path + '_merge.vrt'

    # Nodata note: If the default value of 255 is used we can't look at the images
    #              using stereo_gui.  For now not using a nodata value!

    # TODO: This won't work well if all of the label files go in one folder!
    # Merge all of the label files into a single file
    cmd = 'gdalbuildvrt -vrtnodata None ' + merge_path + ' ' + os.path.join(options.label_folder, '*INWM.tif')
    print(cmd)
    os.system(cmd)
    if not os.path.exists(merge_path):
        print('Failed to run command: ' + cmd)
        return -1

    # Get the projection of the file we want to match
    cmd = 'gdalinfo -proj4 ' + options.landsat_path
    print(cmd)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    out = p.communicate()[0]
    proj_string = None
    lines = out.split('\n')
    for line in lines:
        if '+proj' in line:
            proj_string = line
            break
    if not proj_string:
        raise Exception('Could not read projection string!')

    # Get the projection system coordinates
    (ll_coord, ur_coord) = get_bounding_coordinates(options.landsat_path,
                                                    convert_to_lonlat=False)

    # Reproject the merged label and crop to the landsat extent
    LANDSAT_RES = 30.0 # Meters per pixel
    cmd = ('gdalwarp -overwrite -tr %f %f -t_srs %s -te %s %s %s %s %s %s ' %
           (LANDSAT_RES, LANDSAT_RES, proj_string,
            ll_coord[0], ll_coord[1], ur_coord[0], ur_coord[1],
            merge_path, options.output_path))
    print(cmd)
    os.system(cmd)
    os.remove(merge_path)
    if not os.path.exists(options.output_path):
        print('Failed to run command: ' + cmd)
        return -1


    print('Landsat label file conversion is finished.')
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
