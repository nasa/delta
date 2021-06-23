#!/bin/bash

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

# This is the default SNAP install location
export PATH=~/snap/bin:$PATH
gptPath="gpt"

# Get input parameters
sourceFile="$1"
targetFile="$2"

thisFolder="$PWD/delta/extensions/sources"

# Execute the simple graph

#${gptPath} $thisFolder/sentinel1_default_snap_preprocess_graph.xml -e -PtargetProduct=${targetFile} ${sourceFile}


# Use graph from here:

# Filipponi, F. (2019). Sentinel-1 GRD Preprocessing Workflow. In 
# Multidisciplinary Digital Publishing Institute Proceedings (Vol. 18, No. 1, p. 11).

# https://github.com/ffilipponi/Sentinel-1_GRD_preprocessing

${gptPath} $thisFolder/sentinel1_ffilipponi_snap_preprocess_graph.xml -e -Poutput=${targetFile} -Pinput=${sourceFile} -Pfilter='None' -Presolution=10.0 -Porigin=5.0 -Pdem='SRTM 1Sec HGT' -Pfilter='None' -Pcrs='GEOGCS["WGS84(DD)", DATUM["WGS84", SPHEROID["WGS84", 6378137.0, 298.257223563]], PRIMEM["Greenwich", 0.0], UNIT["degree", 0.017453292519943295], AXIS["Geodetic longitude", EAST], AXIS["Geodetic latitude", NORTH]]'
