#!/usr/bin/env python

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

#pylint: disable=R0915,R0914,R0912
"""
Script to fetch all of the flood images from the USGS HDDS website.
"""
import os
import sys
import argparse
import pickle
import json

# WARNING: In order for this script to work, the api.py file in this module
#          must be modified so it uses the download URL:
#          "https://hddsexplorer.usgs.gov/inventory/json/v/1.4.0"
#          instead of what it normally uses (USGS_API)

from usgs import api


if sys.version_info < (3, 0, 0):
    print('\nERROR: Must use Python version >= 3.0.')
    sys.exit(1)

CATALOG = 'HDDS' # As opposed to EE

#------------------------------------------------------------------------------

def get_dataset_list(options):
    """Return a list of all available HDDS datasets, each entry is (datasetName, pretty name)"""

    dataset_cache_path = os.path.join(options.output_folder, 'dataset_list.csv')
    name_list = []
    if not os.path.exists(dataset_cache_path) or options.refetch_datasets:

        # Each event is a dataset, start by fetching the list of all HDDS datasets.
        print('Submitting HDDS dataset query...')
        results = api.datasets("", CATALOG)
        print(results)
        if not results['data']:
            raise Exception('Did not find any HDDS data!')
        print('Found ' + str(len(results['data'])) + ' matching datasets.')

        # Go through all the datasets and identify the events we are interested in.
        TARGET_TYPES = ['flood', 'hurricane', 'cyclone', 'tsunami', 'dam_collapse', 'storm']
        SKIP = ['test', 'icestorm', 'snowstorm', 'adhoc', 'ad hoc', 'ad_hoc'] # TODO: What is ad hoc here?

        handle = open(dataset_cache_path, 'w')

        for ds in results['data']:

            full_name = ds['datasetFullName'].lower()

            bad = False
            for s in SKIP:
                if s in full_name:
                    bad = True
                    break
            if bad:
                continue

            target = False
            for t in TARGET_TYPES:
                if t in full_name:
                    target = True
                    break
            if not target:
                continue

            if not ds['supportDownload']:
                continue

            print(ds['datasetName'] + ',' + full_name)
            handle.write(ds['datasetName'] + ',' + ds['datasetFullName'] + '\n')
            name_list.append((ds['datasetName'], ds['datasetFullName']))
        handle.close()

    else:
        # Cache exists, load the dataset list from the cache
        with open(dataset_cache_path, 'r') as handle:
            for line in handle:
                parts = line.strip().split(',')
                print(parts)
                name_list.append(parts)

    return name_list

def get_dataset_fields(dataset_list):
    """Code to look through available fields for datasets"""

    for (dataset, _) in dataset_list: #pylint: disable=W0612

        # Get the available filters for this data set
        print('----->  For DS = ' + dataset)
        result = api.dataset_fields(dataset, CATALOG)

        if not result or ('data' not in result):
            print('Failed to get dataset fields for ' + dataset)
            continue

        # Make sure the fields we want to filter on are there
        DESIRED_FIELDS = ['agency - platform - vendor']

        found_count = 0
        for field in result['data']:
            print(field['name'])
            name = field['name'].lower()
            for df in DESIRED_FIELDS:
                if df in name:
                    found_count += 1
                    break
        if found_count < len(DESIRED_FIELDS):
            print('Did not find all desired filter fields!')

        continue

def main(argsIn): #pylint: disable=R0914,R0912

    try:

        usage  = "usage: fetch_hdds_images.py [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--output-folder", dest="output_folder", required=True,
                            help="Download files to this folder.")

        parser.add_argument("--user", dest="user", required=True,
                            help="User name for EarthExplorer website.")
        parser.add_argument("--password", dest="password", required=True,
                            help="Password name for EarthExplorer website.")

        parser.add_argument("--force-login", action="store_true",
                            dest="force_login", default=False,
                            help="Don't reuse the cached EE API key if present.")

        parser.add_argument("--refetch-datasets", action="store_true",
                            dest="refetch_datasets", default=False,
                            help="Force a refetch of the dataset list.")

        parser.add_argument("--refetch-scenes", action="store_true",
                            dest="refetch_scenes", default=False,
                            help="Force refetches of scene lists for each dataset.")

        parser.add_argument("--image-list-path", dest="image_list_path", default=None,
                            help="Path to text file containing list of image IDs to download, one per line.")

        parser.add_argument("--event-name", dest="event_name", default=None,
                            help="Only download images from this event.")

        options = parser.parse_args(argsIn)

    except argparse.ArgumentError:
        print(usage)
        return -1

    if options.output_folder and not os.path.exists(options.output_folder):
        os.mkdir(options.output_folder)

    images_to_use = []
    if options.image_list_path:
        with open(options.image_list_path, 'r') as f:
            for line in f:
                images_to_use.append(line.strip())

    # Only log in if our session expired (ugly function use to check!)
    if options.force_login or (not api._get_api_key(None)): #pylint: disable=W0212
        print('Logging in to USGS EarthExplorer...')
        api.login(options.user, options.password,
                  save=True, catalogId=CATALOG) #pylint: disable=W0612

        print(api._get_api_key(None)) #pylint: disable=W0212

    # Retrieve all of the available datasets
    dataset_list = get_dataset_list(options)

    print('Found ' + str(len(dataset_list)) + ' useable datasets.')

    # Don't think we need to do this!
    #get_dataset_fields(dataset_list)

    counter = 0
    for (dataset, full_name) in dataset_list:
        counter = counter + 1

        if options.event_name: # Only download images from the specified event
            if options.event_name.lower() not in full_name.lower():
                continue

        dataset_folder  = os.path.join(options.output_folder, full_name)
        scene_list_path = os.path.join(dataset_folder, 'scene_list.dat')
        done_flag_path  = os.path.join(dataset_folder, 'done.flag')
        if not os.path.exists(dataset_folder):
            os.mkdir(dataset_folder)

        if os.path.exists(done_flag_path) and not options.refetch_scenes:
            print('Skipping completed dataset ' + full_name)
            continue

        print('--> Search scenes for: ' + full_name)

        BATCH_SIZE = 10000
        if not os.path.exists(scene_list_path) or options.refetch_scenes:
            # Request the scene list from USGS
            #details = {'Agency - Platform - Vendor':'WORLDVIEW', 'Sensor Type':'MS'}
            #details = {'sensor_type':'MS'}
            details = {} # TODO: How do these work??

            # Large sets of results require multiple queries in order to get all of the data
            done  = False
            error = False
            all_scenes = [] # Acculumate all scene data here
            while not done:
                print('Searching with start offset = ' + str(len(all_scenes)))
                results = api.search(dataset, CATALOG, where=details,
                                     max_results=BATCH_SIZE,
                                     starting_number=len(all_scenes), extended=False)

                if 'results' not in results['data']:
                    print('ERROR: Failed to get any results for dataset: ' + full_name)
                    error = True
                    break
                if len(results['data']['results']) < BATCH_SIZE:
                    done = True
                all_scenes += results['data']['results']

            if error:
                continue

            results['data']['results'] = all_scenes

            # Cache the results to disk
            with open(scene_list_path, 'wb') as f:
                pickle.dump(results,f)

        else: # Load the results from the cache file
            with open(scene_list_path, 'rb') as f:
                results = pickle.load(f)

        print('Got ' + str(len(results['data']['results'])) + ' scene results.')

        for scene in results['data']['results']:

            fail = False
            REQUIRED_PARTS = ['displayId', 'summary', 'entityId', 'displayId']
            for p in REQUIRED_PARTS:
                if (p not in scene) or (not scene[p]):
                    print('scene object is missing element: ' + p)
                    print(scene)
                    fail = True
            if fail:
                continue

            # If image list was provided skip other image names
            if images_to_use and (scene['displayId'] not in images_to_use):
                continue

            # Figure out the downloaded file path for this image
            file_name   = scene['displayId'] + '.zip'
            output_path = os.path.join(dataset_folder, file_name)
            if not os.path.exists(dataset_folder):
                os.mkdir(dataset_folder)
            if os.path.exists(output_path):
                continue # Already have the file!

            # Check if this is one of the sensors we are interested in.
            DESIRED_SENSORS = [('worldview','hp'), ('worldview','msi')] # TODO: Add more
            parts = scene['summary'].lower().split(',')
            platform = None
            sensor   = None
            for part in parts:
                if 'platform:' in part:
                    platform = part.split(':')[1].strip()
                if 'sensor:' in part:
                    sensor = part.split(':')[1].strip()
            if (not platform) or (not sensor):
                raise Exception('Unknown sensor: ' + scene['summary'])
            if (platform,sensor) not in DESIRED_SENSORS:
                print((platform,sensor))
                print('Undesired sensor: ' + scene['summary'])
                continue


            # Investigate the number of bands
            PLATFORM_BAND_COUNTS = {'worldview':8, 'TODO':1}
            min_num_bands = PLATFORM_BAND_COUNTS[platform]
            num_bands = None
            try:
                meta = api.metadata(dataset, CATALOG, scene['entityId'])
            except json.decoder.JSONDecodeError:
                print('Error fetching metadata for dataset = ' + dataset +
                      ', entity = ' + scene['entityId'])
                continue
            try:
                for m in meta['data'][0]['metadataFields']:
                    if m['fieldName'] == 'Number of bands':
                        num_bands = int(m['value'])
                        break
                if not num_bands:
                    raise KeyError() # Treat like the except case
                if num_bands < min_num_bands:
                    print('Skipping %s, too few bands: %d' % (scene['displayId'], num_bands))
                    continue
            except KeyError:
                print('Unable to perform metadata check!')
                print(meta)

            # Make sure we know which file option to download
            try:
                types = api.download_options(dataset, CATALOG, scene['entityId'])
            except json.decoder.JSONDecodeError:
                print('Error decoding download options!')
                continue

            if not types['data'] or not types['data'][0]:
                raise Exception('Need to handle types: ' + str(types))
            ready = False
            download_type = 'STANDARD' # TODO: Does this ever change?
            for o in types['data'][0]['downloadOptions']:
                if o['available'] and o['downloadCode'] == download_type:
                    ready = True
                    break
            if not ready:
                raise Exception('Missing download option for scene: ' + str(types))


            # Get the download URL of the file we want.
            r = api.download(dataset, CATALOG, [scene['entityId']],
                             product=download_type)
            try:
                url = r['data'][0]['url']
            except Exception as e:
                raise Exception('Failed to get download URL from result: ' + str(r)) from e

            print(scene['summary'])
            # Finally download the data!
            cmd = ('wget "%s" --user %s --password %s -O %s' %
                   (url, options.user, options.password, output_path))
            print(cmd)
            os.system(cmd)

        print('Finished processing dataset: ' + full_name)
        os.system('touch ' + done_flag_path) # Mark this dataset as finished

        if not os.path.exists(output_path):
            print('ERROR: Failed to download file ' + output_path)

    print('Finished downloading HDDS! files.')
    # Can just let this time out
    #api.logout()

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
