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
Script to extract a list of associated image files from the label file csv list.
"""
import sys


def main(argsIn): #pylint: disable=R0914,R0912

    if len(argsIn) != 2:
        print("usage: convert_image_list.py <input_path> <output_path>")
        return -1

    input_path  = argsIn[0]
    output_path = argsIn[1]

    # Just find the image name for every line with a label ID (integer)
    output_list = []
    with open(input_path, 'r') as f:
        for line in f:
            parts = line.split(',')
            try:
                label_num  = int(parts[0]) #pylint: disable=W0612
                image_name = parts[1]
                output_list.append(image_name)
                #print('%s -> %s' % (label_num, image_name))
            # Header lines etc will throw exceptions trying to cast the integer
            except: #pylint: disable=W0702
                pass

    # Write out a text file with all of the image names.
    with open(output_path, 'w') as f:
        for line in output_list:
            f.write(line+'\n')
    print('Wrote out ' + str(len(output_list)) + ' items.')

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
