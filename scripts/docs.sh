#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
OUT_DIR=$(readlink -m ${1:-./html})
cd $SCRIPTPATH/..
pdoc3 --html -c show_type_annotations=True delta --force -o $OUT_DIR
