#!/bin/bash
# This script takes any number of tiff files as input
# and saves their shapefiles and resolutions to the output
# file projections.zip

TMP_DIR=`mktemp -d`

for filename in "$@"; do
  b=$(basename $filename .tif)
  pixel_size=$(gdalinfo $filename | sed -n -e 's/^Pixel Size = (\(.*\),\(.*\))/\1 \2/p')
  gdaltindex $TMP_DIR/$b.shp $filename > /dev/null
  echo "$pixel_size" > $TMP_DIR/$b.res
done

OUTPUT_ZIP=$PWD/projections.zip
rm -f $OUTPUT_ZIP

cd $TMP_DIR
zip --quiet $OUTPUT_ZIP *.shp *.res

rm -r $TMP_DIR

echo "Saved projections to projections.zip."
