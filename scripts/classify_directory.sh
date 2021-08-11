#!/bin/bash
# This script classifies all tiff images in a directory, preserving the
# directory structure. It also copies any .txt files in the input directory
# to the output. Images that have already been classified are skipped.

INPUT_DIR=.
OUTPUT_DIR=.
PREFIX=IF_

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -i|--input)
    INPUT_DIR="$2"
    shift # past argument
    shift # past value
    ;;
    -o|--output)
    OUTPUT_DIR="$2"
    shift
    shift
    ;;
    -h|--help)
    echo "Usage: $0 -i input_dir -o output_dir --config cfg.yaml [ delta classify arguments ] network_file.h5"
    exit 0
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

INPUT_DIR=$(realpath $INPUT_DIR)
OUTPUT_DIR=$(realpath $OUTPUT_DIR)

# make list of files that haven't been generated yet
TEMP_FILE=$(mktemp)
PROCESS_FILE=$(mktemp)
find $INPUT_DIR \( -name '*.tiff' \) -print > $TEMP_FILE
while IFS="" read -r p || [ -n "$p" ]
do
  OUT_PATH=$OUTPUT_DIR/$(realpath $p -s --relative-to $INPUT_DIR)
  OUT_PATH=$(realpath -m $(dirname $OUT_PATH)/${PREFIX}$(basename $OUT_PATH))
  if [[ ! -f $OUT_PATH ]] ; then
    echo $p >> $PROCESS_FILE
  fi
done < $TEMP_FILE
rm $TEMP_FILE

#copy txt files
OLD_DIR=$PWD
cd $INPUT_DIR
find . \( -name '*.txt' \) -exec cp --parents -t $OUTPUT_DIR {} +
cd $OLD_DIR

# run classification
delta classify --image-file-list $PROCESS_FILE --outdir $OUTPUT_DIR --basedir $INPUT_DIR --outprefix $PREFIX --prob --overlap 32 $@

# TODO: run presoak
presoak --max-cost 20 --elevation $DATA_DIR/fel.vrtd/orig.vrt  --flow $DATA_DIR/p.vrtd/arcgis.vrt   --accumulation  $DATA_DIR/ad8.vrtd/arcgis.vrt  --image --output_dir


# TODO: run arpan tool

rm $PROCESS_FILE
