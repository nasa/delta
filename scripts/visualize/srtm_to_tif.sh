#!/bin/sh

# (c) 2004 Markus Neteler <neteler itc it>
# script to generate BIL header file from SRTM hgt (for GDAL)
#
# $Date: 2004/08/18 09:01:00 $
# Severe bug fixed 2/2004 (let me know if you find more!)
#
# Aug 2004: modified to accept files from other directories
#           (by H. Bowman)
#
###################################################################################
# derived from:
# ftp://edcsgs9.cr.usgs.gov/pub/data/srtm/Documentation/Notes_for_ARCInfo_users.txt
#     (bugfix: the USGS document was updated silently end of 2003)
#
# ftp://edcsgs9.cr.usgs.gov/pub/data/srtm/Documentation/SRTM_Topo.txt
#  "3.0 Data Formats
#  [...]
#  To be more exact, these coordinates refer to the geometric center of 
#  the lower left pixel, which in the case of SRTM-1 data will be about
#  30 meters in extent."
#
#- SRTM 90 Tiles are 1 degree by 1 degree
#- SRTM filename coordinates are said to be the *center* of the LL pixel.
#       N51E10 -> lower left cell center
#
#  BIL uses *center* of the UL (!) pixel:
#      http://downloads.esri.com/support/whitepapers/other_/eximgav.pdf
#  
#  GDAL uses *corners* of pixels for its coordinates. (?)
#
#- BIL HEADER:
#  http://www.uweb.ucsb.edu/~nico/comp/bil_header.htm
#   ->  WRONG!!!!! <-
#
# Even, if small: SRTM is referenced to EGM96:
# http://earth-info.nima.mil/GandG/wgsegm/egm96.html

if [ $# -lt 1 ] ; then
 echo "Script to make BIL file from SRTM hgt.zip file"
 echo "Usage: srtm_generate_hdr.sh XXYYY.hgt.zip"
 echo ""
 exit
fi

if test ! -f $1 ; then
 echo "File '$1' not found"
 exit
fi

ls -1 $1 | grep zip > /dev/null
if [ $? -ne 0 ] ; then
  echo "$1 is no zip file"
  echo "Usage: srtm_generate_hdr.sh XXYY.hgt.zip"
  exit
fi

FILE=`echo $1 | sed 's+.hgt++g'| sed 's+.zip++g'`
TILE=`echo "$FILE" | sed 's+^.*/++'`

###################  
#let' go:
#N18W077.hgt

#echo "Converting file to BIL..."

LL_LATITUDE=`echo $TILE  | cut -b2-3`
LL_LONGITUDE=`echo $TILE | cut -b5-7`

#are we on the southern hemisphere? If yes, make LATITUDE negative.
NORTH=`echo $TILE  | sed 's+.hgt++g' | cut -b1`
if [ "$NORTH" = "S" ] ; then
   LL_LATITUDE=`echo $LL_LATITUDE | awk '{printf "%.10f", $1 * -1 }'`
fi

#are we west of Greenwich? If yes, make LONGITUDE negative.
EAST=`echo $TILE  | sed 's+.hgt++g' | cut -b4`
if [ "$EAST" = "W" ] ; then
   LL_LONGITUDE=`echo $LL_LONGITUDE | awk '{printf "%.10f", $1 * -1 }'`
fi

# Make Upper Left from Lower Left
ULXMAP=`echo $LL_LONGITUDE | awk '{printf "%.1f", $1}'`
# SRTM90 tile size is 1 deg:
ULYMAP=`echo $LL_LATITUDE  | awk '{printf "%.1f", $1 + 1.0}'`

echo "BYTEORDER M
LAYOUT BIL
NROWS 3601
NCOLS 3601
NBANDS 1
NBITS 16
BANDROWBYTES 7202
TOTALROWBYTES 7202
BANDGAPBYTES 0
NODATA -32768
ULXMAP $ULXMAP
ULYMAP $ULYMAP
XDIM 0.000277777777778
YDIM 0.000277777777778"> $TILE.hdr

#rename data file:
unzip -o $FILE.hgt.zip
cp $TILE.hgt $TILE.bil

#create prj file: To be precisely, we would need EGS96!
echo "GEOGCS["wgs84",DATUM["WGS_1984",SPHEROID["wgs84",6378137,298.257223563],TOWGS84[0.000000,0.000000,0.000000]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]" > $TILE.prj

#echo "Make Geotiff (Lat/Long)..."
gdal_translate -ot Int16 $TILE.bil $TILE.tif
rm -f $TILE.bil $TILE.hdr $TILE.hgt $TILE.prj

#echo "Verify Lat/Long SRTM DEM with:"
#echo "   gdalinfo $TILE.tif"
