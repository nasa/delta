TODO


Current dependencies:

[python]
gdal
psutil
usgs

[other]
ESA Sentinel1 Toolbox (for preprocessing Sentinel1 data)


# Running Example Programs

1. Ensure data is downloaded to $PROJECT_HOME/data/in/toy_data/
2. cd $PROJECT_HOME/bin
3. python chunk_and_tensorflow_test.py --image-path $PROJECT_HOME/data/in/toy_data/landsat/<image.tiff> --output-folder $PROJECT_HOME/data/out/test

And the program should execute.  After execution to see the results using MLFlow:

1. cd $PROJECT_HOME/data/out/
2. mlflow ui

And then in web browser open http://localhost:5000



-- miniconda3 installation instructions (tensorflow v1.13):

conda install numpy
conda install gdal
conda install matplotlib
conda install 'tensorflow=*=mkl*'    <--- CPU version!
conda install -c conda-forge mlflow
pip install psutil
