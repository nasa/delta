#!/usr/bin/env python
#pylint: disable=R0914
"""
Given folders of input image/label files, create a new pair of train/validate
folders which contain symlinks to random non-overlapping subsets of the input files.
"""
import os
import sys
import argparse
import random
import yaml

#------------------------------------------------------------------------------

def get_label_path(image_name, options):
    """Return the label file path for a given input image or throw if it is
       not found at the expected location."""

    label_name = image_name.replace(options.image_extension, options.label_extension)
    label_path = os.path.join(options.label_folder, label_name)
    if not os.path.exists(label_path):
        raise Exception('Expected label file does not exist: ' + label_path)
    return label_path

def main(argsIn):

    try:

        usage  = "usage: random_folder_split [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--image-folder", dest="image_folder", required=True,
                            help="Folder containing the input image files.")
        parser.add_argument("--label-folder", dest="label_folder", default=None,
                            help="Folder containing the input label files.")

        parser.add_argument("--output-folder", dest="output_folder", required=True,
                            help="Folder to put the output image/label folders in.")

        parser.add_argument("--validate-fraction", dest="validate_fraction", default=0.3,
                            type=float, help="Fraction of inputs to use for validation.")

        parser.add_argument("--image-ext", dest="image_extension", default='.zip',
                            help="Extension for image files.")
        parser.add_argument("--label-ext", dest="label_extension", default='.tif',
                            help="Extension for label files.")

        parser.add_argument("--image-limit", dest="image_limit", default=None, type=int,
                            help="Only use this many image files total.")

        parser.add_argument("--file-list-path", dest="file_list_path", default=None,
                            help="Path to text file containing list of image file names to use, one per line.")

        parser.add_argument("--config-file", dest="config_path", default=None,
                            help="Make a copy of this config file with paths changed.  The config " +
                            "file must be fully set up, as only the directory entries will be updated.")

        options = parser.parse_args(argsIn)

    except argparse.ArgumentError:
        print(usage)
        return -1

    # Set up the output image structure, wiping any existing data there.
    out_train_folder   = os.path.join(options.output_folder, 'train')
    out_valid_folder   = os.path.join(options.output_folder, 'validate')
    train_image_folder = os.path.join(out_train_folder, 'images')
    train_label_folder = os.path.join(out_train_folder, 'labels')
    valid_image_folder = os.path.join(out_valid_folder, 'images')
    valid_label_folder = os.path.join(out_valid_folder, 'labels')

    if os.path.exists(options.output_folder):
        os.system('rm -rf ' + options.output_folder)
    os.mkdir(options.output_folder)
    os.mkdir(out_train_folder)
    os.mkdir(out_valid_folder)
    os.mkdir(train_image_folder)
    os.mkdir(valid_image_folder)
    if options.label_folder:
        os.mkdir(train_label_folder)
        os.mkdir(valid_label_folder)

    # Recursively find image files, obtaining the full path for each file.
    input_image_list = [os.path.join(root, name)
                        for root, dirs, files in os.walk(options.image_folder)
                        for name in files
                        if name.endswith((options.image_extension))]

    images_to_use = []
    if options.file_list_path:
        with open(options.file_list_path, 'r') as f:
            for line in f:
                images_to_use.append(line.strip())

    train_count = 0
    valid_count = 0
    for image_path in input_image_list:

        # If an image list was provided skip images which are not in the list.
        image_name = os.path.basename(image_path)
        if images_to_use and (image_name not in images_to_use):
            continue

        # Use for validation or for training?
        use_for_valid = (random.random() < options.validate_fraction)

        # Handle the image file
        if use_for_valid:
            image_dest = os.path.join(valid_image_folder, image_name)
            valid_count += 1
        else:
            image_dest = os.path.join(train_image_folder, image_name)
            train_count += 1
        os.symlink(image_path, image_dest)

        if options.label_folder:  # Handle the label file
            label_path = get_label_path(image_name, options)
            label_name = os.path.basename(label_path)
            if use_for_valid:
                label_dest = os.path.join(valid_label_folder, label_name)
            else:
                label_dest = os.path.join(train_label_folder, label_name)
            os.symlink(label_path, label_dest)

        # Check the image limit if it was specified
        total_count = valid_count + train_count
        if options.image_limit and (total_count >= options.image_limit):
            break

    # Copy config file if provided
    if options.config_path:
        config_name = os.path.basename(options.config_path)
        config_out_path = os.path.join(options.output_folder, config_name)

        try:
            with open(options.config_path, 'r') as f:
                config_data = yaml.load(f, Loader=yaml.FullLoader)

            config_data['images']['directory'] = train_image_folder
            config_data['train']['validation']['images']['directory'] = valid_image_folder

            if options.label_folder:
                config_data['labels']['directory'] = train_label_folder
                config_data['train']['validation']['labels']['directory'] = valid_label_folder

            with open(config_out_path, 'w') as f:
                yaml.dump(config_data, f)
            print('Wrote config file: ' + config_out_path)
        except Exception as e: #pylint: disable=W0703
            print('Failed to copy config file!')
            print(str(e))

    print('Wrote %d train files and %d validation files.' % (train_count, valid_count))
    print('Done splitting input files!')
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
