#!/bin/bash

# takes tensorflow version (default is latest) as argument and builds a docker called "delta"

cd "$(dirname "$0")"/../..

if [ -z $1 ]; then
  docker build -t delta -f scripts/docker/Dockerfile .
else
  docker build -t delta -f scripts/docker/Dockerfile --build-arg TF_VERSION=$1 .
fi
