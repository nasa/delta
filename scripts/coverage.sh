#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd $SCRIPTPATH/..
pytest --cov=delta --cov-report=html --cov-config=${SCRIPTPATH}/.coveragerc
