#!/bin/bash

# usage: project_same.sh projection.tiff in.tiff
# Converts in.tiff to overlap projection.tiff with the same
# resolution.

srs_file=$1
in_file=$2
out_file="proj_$2"

echo "Converting ${in_file} to ${out_file}, using area and resolution of ${srs_file}."

data_type=$(gdalinfo ${in_file} | sed -n -e 's/.*Type=\(.*\),.*/\1/p' | head -1)
num_bands=$(gdalinfo ${in_file} | grep "^Band" | wc -l)
band_arg=$(printf -- '-b 1 %.0s' $(eval echo "{1..$num_bands}"))
empty1_file=$(mktemp /tmp/empty1.tiff)
empty2_file=$(mktemp /tmp/empty2.tiff)
gdal_merge.py -createonly -init "0 0 0" -ot ${data_type} -o ${empty1_file} ${srs_file}
gdal_translate -ot ${data_type} ${band_arg} ${empty1_file} ${empty2_file}
rm ${empty1_file}

pjt_file=$(mktemp /tmp/pjt.XXXXXX)
pjt_img=$(mktemp /tmp/pjt_img.XXXXXX.tiff)
#upper_left=$(gdalinfo ${srs_file} | sed -n -e 's/^Upper Left *( *\(.*\), *\(.*\)).*)/\1 \2/p')
#lower_right=$(gdalinfo ${srs_file} | sed -n -e 's/^Lower Right *( *\(.*\), *\(.*\)).*)/\1 \2/p')
pixel_size=$(gdalinfo ${srs_file} | sed -n -e 's/^Pixel Size = (\(.*\),\(.*\))/\1 \2/p')
shp_file=/tmp/shape.shp # cannot have uppercase for some reason...
gdaltindex ${shp_file} ${srs_file}
gdalsrsinfo -o wkt "${srs_file}" > "${pjt_file}"
gdalwarp -r bilinear -t_srs "${pjt_file}" -tr ${pixel_size} -cutline ${shp_file} -crop_to_cutline "${in_file}" "${out_file}"
rm ${pjt_file} ${shp_file}
rm ${empty2_file}

