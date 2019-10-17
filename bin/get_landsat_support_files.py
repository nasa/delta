#!/usr/bin/python
"""
Script to fetch support images from USGS corresponding to a given Landsat image.
"""
import os
import sys
import argparse
import subprocess
import shutil

import gdal
from osgeo import osr

from usgs import api

from delta.imagery import utilities

#------------------------------------------------------------------------------


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
        if ext != '.zip':
            continue

        name_out = f.replace('.hgt.zip','.tif')

        # Look to see if we have a matching label file
        tar_path   = os.path.join(tar_folder, f)
        label_path = os.path.join(tar_folder, name_out)

        # If we did not find the INWM file, untar.
        if not utilities.file_is_good(label_path):
            #utilities.unpack_to_folder(tar_path, unpack_folder)
            cmd = 'srtm_to_tif.sh ' + tar_path
            print(cmd)
            os.system(cmd)
            try: # srtm_to_tif unpacks to the current folder
                shutil.move(name_out, label_path)
            except FileNotFoundError:
                pass
            # Look again for a matching file
            if not utilities.file_is_good(label_path):
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


def fetch_images(ll_coord, ur_coord, output_folder, options):
    """Download all images that fit the given criteria to the output folder
       if they are not already present.  The coordinates must be in lon/lat degrees.
    """

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    CATALOG = 'EE'
    #DATASET = 'SP_TILE_DSWE'
    product='DSWE'
    DATASET = 'SRTM_V3_SRTMGL1'
    product='STANDARD'

    # Only log in if our session expired (ugly function use to check!)
    if options.force_login or (not api._get_api_key(None)): #pylint: disable=W0212
        print('Logging in to USGS EarthExplorer...')
        dummy_result = api.login(options.user_ee, options.password_ee,
                                 save=True, catalogId=CATALOG)

        #print(api._get_api_key(None))
        #raise Exception('DEBUG')

    print('Submitting EarthExplorer query...')
    results = api.search(DATASET, CATALOG, where={},# start_date=date, end_date=date,
                         ll=dict([('longitude',ll_coord[0]),('latitude',ll_coord[1])]),
                         ur=dict([('longitude',ur_coord[0]),('latitude',ur_coord[1])]),
                         max_results=12, extended=False)

    if not results['data']:
        raise Exception('Did not find any data that matched the Landsat file!')
    print('Found ' + str(len(results['data']['results'])) + ' matching files.')

    for scene in results['data']['results']:
        #print('------------')
        #print(scene)
        print('Found match: ' + scene['entityId'])

        #fname = scene['entityId'] + '.tar'
        fname = scene['displayId'].replace('.SRTMGL1','') + '.hgt.zip'
        output_path = os.path.join(output_folder, fname)

        if utilities.file_is_good(output_path):
            print('Already have image on disk!')
            continue

        r = api.download(DATASET, CATALOG, [scene['entityId']], product=product)
        print('Download response:')
        print(r)
        if not r['data']:
            raise Exception('Failed to get download URL!')
        url = r['data'][0]['url']
        cmd = ('wget "%s" --user %s --password %s -O %s'
               % (url, options.user_urs, options.password_urs, output_path))
        print(cmd)
        os.system(cmd)

        if not utilities.file_is_good(output_path):
            raise Exception('Failed to download file ' + output_path)

    print('Finished downloading files.')
    # Can just let this time out
    #api.logout()



def main(argsIn):

    try:

        usage  = "usage: get_landsat_support_files [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--output-path", dest="output_path", required=True,
                            help="Output image path.")

        parser.add_argument("--landsat-path", dest="landsat_path", required=True,
                            help="Path to the landsat image we want the label to match.")

        parser.add_argument("--dl-folder", dest="label_folder", required=True,
                            help="Download files to this folder.")

        parser.add_argument("--user-ee", dest="user_ee", required=False,
                            help="User name for EarthExplorer website, needed to download new files.")
        parser.add_argument("--password-ee", dest="password_ee", required=False,
                            help="Password for EarthExplorer website, needed to download new files.")

        parser.add_argument("--user-urs", dest="user_urs", required=False,
                            help="User name for NASA URS account, needed to download SRTM files.")
        parser.add_argument("--password-urs", dest="password_urs", required=False,
                            help="Password for NASA URS account, needed to download SRTM files.")

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

    (ll_coord, ur_coord) = get_bounding_coordinates(options.landsat_path,
                                                    convert_to_lonlat=True)

    if options.user_ee and options.password_ee and options.user_urs and options.password_urs:
        print('Login info provided, searching for overlapping label images...')
        fetch_images(ll_coord, ur_coord, options.label_folder, options)
    else:
        print('user and password inputs not provided, skipping label download step.')

    ## Untar the input files if needed
    untar_folder = options.label_folder
    input_files = unpack_inputs(options.label_folder, untar_folder)
    if not input_files:
        print('Did not detect any unpacked files!')
        return -1

    merge_path = options.output_path + '_merge.vrt'

    # Nodata note: If the default value of 255 is used we can't look at the images
    #              using stereo_gui.  For now not using a nodata value!

    # TODO: This won't work well if all of the label files go in one folder!
    # Merge all of the label files into a single file
    cmd = 'gdalbuildvrt -vrtnodata None ' + merge_path + ' ' + os.path.join(options.label_folder, '*.tif')
    print(cmd)
    os.system(cmd)
    if not os.path.exists(merge_path):
        print('Failed to run command: ' + cmd)
        return -1

    # Get the projection and pixel size of the file we want to match
    cmd = 'gdalinfo -proj4 ' + options.landsat_path
    print(cmd)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    out = p.communicate()[0]
    proj_string = None
    x_res = 30 # LANDSAT default, 30 meters per pixel
    y_res = 30
    lines = out.split('\n')
    for line in lines:
        if '+proj' in line:
            proj_string = line
            continue
        if 'Pixel Size' in line:
            start = line.find('(')
            stop  = line.find(')')
            s = line[start+1:stop]
            parts = s.split(',')
            x_res = float(parts[0])
            y_res = float(parts[1])
            continue
    if not proj_string:
        raise Exception('Could not read projection string!')

    # Get the projection system coordinates
    (ll_coord, ur_coord) = get_bounding_coordinates(options.landsat_path,
                                                    convert_to_lonlat=False)

    # Reproject the merged label and crop to the landsat extent
    cmd = ('gdalwarp -overwrite -tr %.20f %.20f -t_srs %s -te %s %s %s %s %s %s ' %
           (x_res, y_res, proj_string,
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
