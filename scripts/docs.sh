#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
OUT_DIR=$(readlink -m ${1:-./html})
cd $SCRIPTPATH/..
rm -r $OUT_DIR/* # keep .git folder in CI
pdoc3 --html --skip-errors -c show_type_annotations=True delta --force -o $OUT_DIR
cp -r docs/ $OUT_DIR/docs
mv $OUT_DIR/delta/* $OUT_DIR/ && rmdir --ignore-fail-on-non-empty $OUT_DIR/delta
