TODO


Current dependencies:

[python]
gdal
psutil
usgs

[other]
ESA Sentinel1 Toolbox (for preprocessing Sentinel1 data)




-- miniconda3 installation instructions (tensorflow v1.13):

conda install numpy
conda install gdal
conda install matplotlib
conda install 'tensorflow=*=mkl*'    <--- CPU version!
conda install -c conda-forge mlflow
pip install psutil
