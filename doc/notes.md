------
title: Deep Earth Learnting, Training, and Analysis (DELTA)
author: P. M. Furlong
------

# 2019-03-18

- Installed tensorflow on fnord using a virtual environment.
- Installed MLFlow on 
- Trained network on MNIST data.

## Installing things:

You need to have SG (if your computer is administered by SG) install python3 and pip3 and the cuda drivers if you are going to be using the

### VirtualEnv

~~~~~ {#install-venv .shell}
$  pip3 install -U --user virtualenv
~~~~~

## Setting up the virtual env

To start a session:

~~~~~ {#start-venv .shell}
$ cd delta
$ source ./venv/bin/activate
~~~~~

At this point you should be in the virtual environment, and can install things using pip as if you were root.  To end the session:


~~~~~ {#end-venv .shell}
(venv) $ deactivate
~~~~~

### Tensorflow

~~~~~ {#install-tensorflow .shell}
(venv) $ pip install tensorflow
~~~~~

## GDAL

~~~~~ {#install-gdal .shell}
(venv) $ pip install --global-option=build_ext --global-option="-I/usr/include/gdal" GDAL==`gdal-config --version`
~~~~~ 


## Interacting with mlflow

You annotate your code with different functions which can log different classes of things (metrics, parameters, artifacts).  

You can view the different things which have been logged by running:

~~~
(venv) $ mlflow ui
~~~ 

This will start a server at http://localhost:5000, which you can view and sort through result.

- Note: MLFlow can produce servers which treat the learned model as a function that can be invoked. Maybe this will make sharing the auto-encoder learned networks easier. 
