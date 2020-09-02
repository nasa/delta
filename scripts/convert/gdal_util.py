# Copyright © 2020, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All rights reserved.
#
# The DELTA (Deep Earth Learning, Tools, and Analysis) platform is
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Diverse useful functions for handeling new datasets using the gdal python bindings:
    - Merging images
    - Matching images
    - Comparing images
    - Computing shared region between images
    - Filling noData pixels

Usual parameters:
    - input_path(s): path(s) to the file(s) that will be processed. Should be a string or a list of string.
    - output_path(/file): path to the directory(/file) where the generated output(s) will be stored
    - db_ref(/new): Gdal datasets obtained with 'gdal.Open(path_to_file)'. Ref is for reference

This file needs more cleanup and testing.
"""

import os

import numpy as np
from scipy import ndimage

from osgeo import gdal, osr, gdalconst

# Indexes for Geotransform metadata
XCOORD = 0
YCOORD = 3
XRES = 1
YRES = 5

def merge_images(input_paths, output_file):
    """
    Merge tiles of an image into a single big image.
    Merging is based on the GeoTransforms of the tiles, input order is irrelevant.
    If some part of the overall scene is not covered by the tiles, it will be filled with
    NoDataValue in the final image. The images are first compared to check if they can be
    merged: Same resolution, nb of bands, projection, ... The result is stored in 'output_file'
    """

    args_command = '-o %s' % (output_file) # arguments for the merge command
    for i, path in enumerate(input_paths):
        if i == 0:
            ref_image = gdal.Open(path)  # the first input image is considered as the reference and will be compared
            ref_path = path              # with all the other images to make sur that merging is possible
        else:
            # Number of difference between two input images. Discarding the comparisons between the size of the bands
            # and the scene coverage because they don't prevent merging
            if not images_equivalent(ref_image, gdal.Open(path), raster_size=False, coverage=False):
                print("Differences between %s and %s prevent merging. Aborted." %
                      (os.path.split(path)[1], os.path.split(ref_path)[1]))
                return
        args_command += ' ' + path

    assert os.system('gdal_merge.py ' + args_command) == 0, 'Merging failed.'
    print("Merged the images successfully in {}".format(output_file))


def match_images(input_paths, output_path, target_path=None):
    """
    Transform the input images to match the target image.
    Matching the GeoTransform (resolution + coverage) and the spatial reference system.
    The data type of the source will stay the same.
    The images are first compared to check if they can be matched directly: need the same scene cover.
    If not, it will try to find the largest shared scene between all the images (inputs + target).
    All images will be cropped to cover only the shared scene.
    The generated images will be stored in a new directory: 'target_name_match' localized in output_path
    """
    target_need_crop = False  # Target needs cropping if it doesn't cover exactly the same scene as the inputs

    if isinstance(input_paths, str):
        input_paths = [input_paths]

    if target_path is None:  # If no target is given, the first input is considered as the target
        if len(input_paths) == 1:
            print("Only one input was given. Need a target or another input to match.")
            return
        target_path = input_paths[0]
        target_name = os.path.split(input_paths[0])[1].split('.')[0]
        input_paths = np.delete(input_paths, 0)
        print("No target was given. Replacing it with the first input: {}".format(target_name))
    else:
        target_name = os.path.split(target_path)[1].split('.')[0]

    data_match = gdal.Open(target_path)
    geo_match = data_match.GetGeoTransform()  # target image characteristics used as reference
    srs_match = data_match.GetSpatialRef()
    corner_match_dict = gdal.Info(data_match, format='json')['cornerCoordinates']
    corner_match_arr = [corner_match_dict.get(key) for key in corner_match_dict.keys()
                        if key in ('upperLeft', 'lowerRight')]

    # sets the biggest shared region to the whole image
    biggest_common_bounds = (corner_match_arr[0][0], corner_match_arr[1][1], corner_match_arr[1][0],
                             corner_match_arr[0][1])

    #  Stores the characteristics of the source images during the first for-loop
    type_source = [None]*len(input_paths)
    srs_source = [None]*len(input_paths)
    output_file = ['']*len(input_paths)

    new_dir = target_name + '_match'
    new_path = os.path.join(output_path, new_dir)
    try:
        os.makedirs(new_path)
    except OSError:
        print('A directory already exists for images matched with {}. Old files can be replaced.\n'.format(target_name))

    # Find biggest common overlapping region to all the images
    for i, path in enumerate(input_paths):
        img = gdal.Open(path)
        input_name, file_ext = os.path.split(path)[1].split('.')
        srs_source[i] = img.GetSpatialRef()
        type_source[i] = img.GetRasterBand(1).DataType
        output_file[i] = os.path.join(new_path, input_name + '_match.' + file_ext)

        # Checks if the images cover the same scene
        if not images_equivalent(data_match, img, projection=False, data_type=False,
                                 num_bands=False, raster_size=False, resolution=False):
            # Compute possible overlapping region
            overlap_geo, overlap_pix_match, overlap_pix_src = compute_overlap_region(data_match, img)

            # No overlap between the images then process is stopped
            if overlap_geo is None or overlap_pix_match is None or overlap_pix_src is None:
                print("Can't match {} to {} because they don't overlap geographically.".format(path,
                                                                                               target_name))
                return
            # Updates the shared region boundaries
            biggest_common_bounds = (max(biggest_common_bounds[0], overlap_geo[0][0]),
                                     max(biggest_common_bounds[1], overlap_geo[1][1]),
                                     min(biggest_common_bounds[2], overlap_geo[1][0]),
                                     min(biggest_common_bounds[3], overlap_geo[0][1]))

            # Checks if the previous biggest shared region and the new computed shared region overlaps. If not, process
            # is stopped
            if biggest_common_bounds[0] > biggest_common_bounds[2] \
                    or biggest_common_bounds[1] > biggest_common_bounds[3]:
                print("Some inputs do not have a common overlapping region with the target. Matching them is not "
                      "possible (Problem occured for {}).".format(path))
                return

            if corner_match_arr != overlap_geo:
                target_need_crop = True

        else:
            print("{} and {} cover the same region (left corner:[{}, {}], right corner:[{}, {}])."
                  .format(input_name, target_name, corner_match_arr[0][0], corner_match_arr[0][1],
                          corner_match_arr[1][0], corner_match_arr[1][1]))

    print("\nCommon overlapping region to all the images: {}".format(biggest_common_bounds))

    for i, path in enumerate(input_paths):

        print("Matching {} to {} in the overlapping region {}.".format(path, target_name,
                                                                       biggest_common_bounds))

        # Actual matching operation. Resampling algorithm is bilinear. Only matches the shared region of the images
        gdal.Warp(output_file[i], path, dstSRS=srs_match, srcSRS=srs_source[i], outputType=type_source[i],
                  xRes=geo_match[XRES], yRes=geo_match[YRES], resampleAlg=gdalconst.GRIORA_Bilinear,
                  outputBounds=biggest_common_bounds)

    if target_need_crop:
        crop_target_path = os.path.join(new_path, target_name + '_crop.' + file_ext)

        print("Cropping the overlapping region from the target so that the images cover the same region.")

        # Crops the target if needed to only cover the shared region with the inputs
        gdal.Translate(crop_target_path, target_path, projWin=[biggest_common_bounds[0], biggest_common_bounds[3],
                                                               biggest_common_bounds[2], biggest_common_bounds[1]])

    return


def images_equivalent(db_ref, db_new, projection=True, data_type=True, num_bands=True, raster_size=True,
                      resolution=True, coverage=True):
    """
    Compare some characteristics of two images (needs to be gdal datasets):
        - Projections and Spatial reference system (SRS): Can be different but equivalent.
        - Type of data: Usually Bytes, float, uintXX, ...
        - Number of bands
        - Bands' dimensions: equivalent to the shape of the image (in Pixels)
        - Spatial resolutions: X and Y axis
        - Coverage: Corner coordinates of the image in the SRS

    inputs: Which characteristic not to compare can be specified in the kwargs (by default: compare all characteristics)
    outputs: True if characteristics are the same, false otherwise
    """
    # pylint: disable=too-many-return-statements

    assert isinstance(db_ref, gdal.Dataset) and isinstance(db_new, gdal.Dataset), 'Inputs must be gdal datasets.'

    # Compares Projections and SRS
    if projection:
        if db_new.GetProjection() != db_ref.GetProjection():
            # Checks if the projections are equivalent eventhough there are not exactly the same
            if not osr.SpatialReference(db_ref.GetProjection()).IsSame(osr.SpatialReference(db_new.GetProjection())):
                return False

    # Compares data type
    if data_type:
        dtype_ref = gdal.GetDataTypeName(db_ref.GetRasterBand(1).DataType)
        dtype_new = gdal.GetDataTypeName(db_new.GetRasterBand(1).DataType)

        if dtype_new != dtype_ref:
            return False

    # Compares number of bands
    if num_bands and db_ref.RasterCount != db_new.RasterCount:
        return False

    # Compares the bands' dimension
    if raster_size:
        gSzX = db_ref.RasterXSize
        nSzX = db_new.RasterXSize
        gSzY = db_ref.RasterYSize
        nSzY = db_new.RasterYSize

        if gSzX != nSzX or gSzY != nSzY:
            return False

    # Compares the spatial resolution
    if resolution:
        geo_ref = db_ref.GetGeoTransform()
        geo_new = db_new.GetGeoTransform()

        if geo_ref[XRES] != geo_new[XRES] or geo_ref[YRES] != geo_new[YRES]:
            return False

    # Compares the coverage
    if coverage:
        cornerCoord_ref = gdal.Info(db_ref, format='json')['cornerCoordinates']
        cornerCoord_new = gdal.Info(db_new, format='json')['cornerCoordinates']

        if cornerCoord_ref != cornerCoord_new:
            return False

    return True

def compute_overlap_region(db_ref, db_new):
    """
    Computes the overlapping/shared region between two images.
    Outputs:
        - Corner coordinates of the overlapping region in the SRS
        - Corresponding pixel indexes in both input images
    """

    cornerCoord_ref = gdal.Info(db_ref, format='json')['cornerCoordinates']
    cornerCoord_new = gdal.Info(db_new, format='json')['cornerCoordinates']

    cc_val_ref = [cornerCoord_ref.get(key) for key in cornerCoord_ref.keys() if key != 'center']
    [ul_ref, dummy, lr_ref, _] = cc_val_ref

    cc_val_new = [cornerCoord_new.get(key) for key in cornerCoord_new.keys() if key != 'center']
    [ul_new, dummy, lr_new, _] = cc_val_new

    # Checks if the images cover exactly the same regions.
    if cc_val_ref == cc_val_new:
        print("The images cover the same region (left corner:[{}, {}], right corner:[{}, {}])."
              .format(ul_ref[0], ul_ref[1], lr_ref[0], lr_ref[1]))
        return [ul_ref, lr_ref], [[0, 0], [db_ref.RasterXSize, db_ref.RasterYSize]], \
               [[0, 0], [db_ref.RasterXSize, db_ref.RasterYSize]]

    # Computes the overlapping region
    overlap_corners = [[max(ul_ref[0], ul_new[0]), min(ul_ref[1], ul_new[1])],[min(lr_ref[0], lr_new[0]),
                                                                               max(lr_ref[1], lr_new[1])]]

    # Checks if the overlapping region is physically possible. If not, then the images don't cover the same region
    if overlap_corners[0][0] > overlap_corners[1][0] or overlap_corners[0][1] < overlap_corners[1][1]:
        print("The two regions represented by the images don't overlap.")
        return None, None, None
    print("Found an overlapping regions (left corner:[{}, {}], right corner:[{}, {}])."
          .format(overlap_corners[0][0], overlap_corners[0][1], overlap_corners[1][0], overlap_corners[1][1]))

    # If a shared region is found then compute the pixels indexes corresponding to its corner coordinates in
    # for both images
    col_ul_ref, row_ul_ref = geo2pix(db_ref.GetGeoTransform(), overlap_corners[0][0], overlap_corners[0][1])
    col_lr_ref, row_lr_ref = geo2pix(db_ref.GetGeoTransform(), overlap_corners[1][0], overlap_corners[1][1])

    col_ul_new, row_ul_new = geo2pix(db_new.GetGeoTransform(), overlap_corners[0][0], overlap_corners[0][1])
    col_lr_new, row_lr_new = geo2pix(db_new.GetGeoTransform(), overlap_corners[1][0], overlap_corners[1][1])

    return overlap_corners, [[row_ul_ref, col_ul_ref], [row_lr_ref, col_lr_ref]], [[row_ul_new, col_ul_new],
                                                                                   [row_lr_new, col_lr_new]]


def fill_no_data_value(input_path, interpolation_distance=16, create_new_file=True,
                       output_file=None, no_data_value=None):
    """
    Fill noData pixels by interpolation from valid pixels around them.
    inputs:
        - Interpolation_distance: Maximum number of pixels to search in all
          directions to find values to interpolate from.
        - create_new_file: If True, the filled image will be stored in output_file
          instead of modifying the original file.
        - output_file: Where the new file is stored
        - no_data_value: Not useful in most cases. If the noDataValue of the gdal dataset is not set.
    This algorithm is generally suitable for interpolating missing regions of fairly continuously varying rasters
    (such as elevation models for instance).
    It is also suitable for filling small holes and cracks in more irregularly varying images (like airphotos).
    It is generally not so great for interpolating a raster from sparse point data.
    Returns 0 if a filling was necessary and 1 if all the pixels were already valid.
    """
    # create mask file or not
    # gdal.SetConfigOption('GDAL_TIFF_INTERNAL_MASK', 'NO')

    # Get driver to write the output file
    file_format = "GTiff"
    driver = gdal.GetDriverByName(file_format)
    input_head, input_tail = os.path.split(input_path)
    input_name, input_ext = input_tail.split('.')

    # Opens dataset in update mode
    db = gdal.Open(input_path, gdal.GA_Update)

    # Creates new file if instructed
    if create_new_file:
        if output_file is None:
            print("A path for the output file is mandatory if create_new_file is True. Because none was given, "
                  "the new file will be saved at the same location as the input.")
            output_file = os.path.join(input_head, input_name + '_filled.' + input_ext)
        new_db = driver.CreateCopy(output_file, db)


    # Else, updates the original image
    else:
        new_db = db

    doFill = [True]*db.RasterCount

    # Checks if each band needs filling
    for i in range(0, new_db.RasterCount):
        band = new_db.GetRasterBand(i+1)

        # If needed, sets the value of the noData pixels
        if no_data_value is not None and band.GetNoDataValue() is None:
            band.SetNoDataValue(no_data_value)

        # Check the gdal dataset flag. The band won't be filled if flag == GMF_ALL_VALID
        if band.GetMaskFlags() == gdal.GMF_ALL_VALID:
            print("All pixels of band {} of {} have a value. No need to fill it.".format(i + 1, input_name))
            doFill[i] = False

    # If all bands don't need to be filled then the process is stopped
    if np.unique(doFill).size == 1 and not np.unique(doFill)[0]:
        print("All pixels of {} are valid. No filling will be applied.".format(input_name))
        return 0

    # Fills each band and applies once a 3x3 smoothing filter on the filled areas
    for i in range(0, new_db.RasterCount):

        if doFill[i]:
            band = new_db.GetRasterBand(i+1)
            gdal.FillNodata(band, None, interpolation_distance, 1)
    print("Filled successfully the pixels without data of {}.".format(input_tail))
    return 1


def pix2geo(geo_transform, xpix, ypix):
    """
    Computes the coordinate in the spatial reference system of a pixel given its indexes in the image array and the
    geotransform of the latter.
    """

    xcoord = xpix*geo_transform[1] + geo_transform[0]
    ycoord = ypix*geo_transform[5] + geo_transform[3]

    return xcoord, ycoord


def geo2pix(geo_transform, xcoord, ycoord):
    """
    Computes the indexes of the pixel in the image array corresponding to a point with given coordinates in the
    spatial reference system.
    """

    xpix = int((xcoord-geo_transform[0])/geo_transform[1])
    ypix = int((ycoord - geo_transform[3]) / geo_transform[5])

    return xpix, ypix


def compute_normalized_dsm(dsm_file, dem_file, apply_blur=False, output_file=None):
    """
    Computes the normalized Digital Surface Model (nDSM) from a DSM and its corresponding Digital Terrain Model (DTM).
    The nDSM is obtained by simply subtracting the DTM to the DSM (pixel-wise).
    If apply_blur is True then a blurring filter is applied to the DEM prior to the computation.
    """

    # Get driver to write the output file
    file_format = "GTiff"
    driver = gdal.GetDriverByName(file_format)

    # If no output is given then creates the nDSM where the DSM is stored
    if output_file is None:
        print('No output file was given. The nDSM will be stored where the DSM is: {}'
              .format(os.path.split(dsm_file)[0]))
        output_file = os.path.join(os.path.split(dsm_file)[0], 'nDSM.tif')

    # Applies blurring effect on the DEM (keeps the original)
    if apply_blur:
        db_dem = gdal.Open(dem_file)
        path, ext = dem_file.split('.')

        # Creates copy of the DEM to store the blurred data
        blur_file = path + '_blurred.' + ext
        db_blur = driver.CreateCopy(blur_file, db_dem)
        band = db_blur.GetRasterBand(1)
        data = band.ReadAsArray()
        blurred_data = ndimage.gaussian_filter(data, 9)
        band.WriteArray(blurred_data)
        dem_file = blur_file

    # Subtraction of the two rasters
    assert os.system('gdal_cacl.py --calc=A-b -A %s -B %s --outfile %s --quiet' % \
                     (dsm_file, dem_file, output_file)) == 0

    print('Computed the normalized dsm successfully.')
