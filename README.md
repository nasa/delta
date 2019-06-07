# DELTA

DELTA (Deep Earth Learning, Tools, and Analysis) is a framework for deep learning on satellite imagery.

It is currently under active development.

## Installation

1. Run ./scripts/setup.sh to install DELTA's dependencies in Ubuntu.

Note that DELTA uses python3.

## Running Example Programs

1. Download landsat data to $PROJECT_HOME/data/in/toy_data/
3. python $PROJECT_HOME/bin/chunk_and_tensorflow_test.py --image-path $PROJECT_HOME/data/in/toy_data/landsat/<image.tiff> --output-folder $PROJECT_HOME/data/out/test

And the program should execute.  After execution to see the results using MLFlow:

1. cd $PROJECT_HOME/data/out/
2. mlflow ui
3. Open a web browser to http://localhost:5000

## Scott's Old Install Directions (don't use)

-- miniconda3 installation instructions (tensorflow v1.13):
-- (for Scott's use only)

conda install numpy
conda install gdal
conda install matplotlib
conda install 'tensorflow=*=mkl*'    <--- CPU version!
conda install -c conda-forge mlflow
pip install psutil

