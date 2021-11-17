#!/bin/bash

# assumes build.sh has already been run.
# Mounts the current working directory in docker and then runs the given command in the container.
# Note that this is very insecure

docker run -it --rm --mount source=`pwd`,target=/volume,type=bind -w /volume delta "$@"
