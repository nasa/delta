"""
@author P.M. Furlong
@date 2019.03.21
@see https://github.com/rcampbell95/MODIS-Classifier
"""

import mlflow
import numpy as np
import math
import random

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import os
import sys

import gdal

def load_data(bands, path):
    """Load multispectral image data"""
    dirpath, dirname, filenames = next(os.walk(path))


    data = []
    for file in filenames:
        bands = []
        image = gdal.Open(dirpath + "/" + file)
        num_bands = image.RasterCount

        for band_i in range(num_bands):
            rband = image.GetRasterBand(band_i + 1)
            # Fill nans in data
            bands.append(np.nan_to_num(rband.ReadAsArray()))
        ### end for
        data.append(bands)
    ### end for
        
    return np.array(data), filenames
### end load_data

def merge_dims(image):
	"""Create an ndarray that merges the dimensions of the subarrays in 
	an ndarray"""
# 	import numpy as np
	
	shape = image.shape + image[0].shape
	new_image = np.ndarray(shape=shape)
	
	for i in range(image.shape[-1]):
		new_image[i] = image[i]

	return new_image
### end merge_dims

def chunk_image(data, k=3, label=False):
    """
    Break a multispectral image into (7,k,k) tensors

    @param data The image which is being broken into square chunks, with dimensions 
        CxHxW where in this application C is channels, H is height, and W is width

    @param k The size of the chunks

    @param label Determines whether or not this is source data or the labels for the data.
    
    @returns An array which is [number of chunks, number of channels, k, k] in shape.
    """
    if label == True:
        try:
            height = data.shape[0]
            width = data.shape[1]
        except:
            print(data.shape)
    else:
        height = data.shape[1]
        width = data.shape[2]
    ### end if
    
    batches = (height - k + 1) * (width - k + 1)
    
    if label == True:
        shape = (batches)
    else:
        # TODO: Check why this was 9 and not 7
        shape = (batches, 7, k, k)
    	
    out = np.zeros(shape=shape)
    
    offset = (k - 1) // 2
    if label == True:
        
        out = data[offset:height - offset, offset:width - offset]
        out = np.ravel(out)
    else:
        batch = 0
        
        for h in range(0, height - 2):
            for w in range(0, width - 2):
                out[batch,:,:,:] = data[0:7, h:h + 3, w:w + 3]
                batch += 1
            ### end for
        ### end for
            
    return (out,width-2*offset, height-2*offset)
### end chunk_data


def get_data(filename):
    # Load Data
    data, filenames = load_data(None,os.path.abspath(filename))

    labels = [data[i][7] for i in range(len(data))]
    train_data = [data[i][:7] for i in range(len(data))]
    return labels, train_data, filenames

def make_model(channel, in_len):
    fc1_size = channel * in_len ** 2
#     fc2_size = fc1_size * 2
#     fc3_size = fc2_size
#     fc4_size = fc1_size
    # To be consistent with Robert's poster
    fc2_size = 162#253
    fc3_size = 162#253
    fc4_size = 40#81

    dropout_rate = 0.3 # Taken from Robert's code.

    mlflow.log_param('fc1_size',fc1_size)
    mlflow.log_param('fc2_size',fc2_size)
    mlflow.log_param('fc3_size',fc3_size)
    mlflow.log_param('fc4_size',fc4_size)
    mlflow.log_param('dropout_rate',dropout_rate)
    

    # Define network
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(channel, in_len, in_len)),
        # Note: use_bias is True by default, which is also the case in pytorch, which Robert used.
        keras.layers.Dense(fc2_size, activation=tf.nn.relu), 
        keras.layers.Dropout(rate=dropout_rate),
        keras.layers.Dense(fc3_size, activation=tf.nn.relu),
        keras.layers.Dropout(rate=dropout_rate),
        keras.layers.Dense(fc4_size, activation=tf.nn.relu),
        keras.layers.Dropout(rate=dropout_rate),
        keras.layers.Dense(1, activation=tf.nn.sigmoid),
        ])
    return model

def split_trainset(train_data, train_labels, ratio, batch_size):
    """Split train data into train and validation set
    with given ratio"""
    
    data = list(zip(train_data, train_labels))
    
    random.shuffle(data)
    
    train_val_data, train_val_labels = list(zip(*data))
    
    train_ratio = ratio
    
    trainsize = int(len(train_val_data) * train_ratio)

    train_x = np.array(train_val_data[:trainsize])
    train_y = np.array(train_val_labels[:trainsize])

    val_x = np.array(train_val_data[trainsize:])
    val_y = np.array(train_val_labels[trainsize:])
	
    return (train_x,train_y),(val_x,val_y) 


def main(channel, in_len,seed_val=470127,img_num=0,experiment_id='Default'):
    
    random.seed(123541)
    tf.random.set_random_seed(seed_val)
    # Start mlflow logging
    mlflow.set_tracking_uri('file:./mlruns')
    mlflow.set_experiment(experiment_id)
    mlflow.start_run()

    params = {}
    artifacts = []
    metrics = {}

    print('Getting data')
    data_dir = '../data/in/toy_data/landsat'
    if not os.path.exists(data_dir):
        print('Looking for data in %s, but it does not exist')
        exit()
    labels, train_data, filenames = get_data('../data/in/toy_data/landsat')

    # Chunk Images
    print('{:30} shape: (batch, channel, height, width)'.format('filename'))

#     img_num = 0 
    # Image number
    params['train_image'] = filenames[img_num]
    (chunked_data,width,height) = chunk_image(merge_dims(train_data[img_num]))
    (chunked_labels,width,height) = chunk_image(labels[img_num], label=True)
    print('labels shape: ', chunked_labels.shape)

    batch_size = 2048
    params['batch_size'] = batch_size
    mlflow.log_param('batch_size', batch_size)
    mlflow.log_param('image_num',img_num)
    mlflow.log_param('training_img',filenames[img_num])

    print('Splitting training data')
    split_ratio = 0.7
    mlflow.log_param('split_ratio',split_ratio)
    (train_x,train_y),(val_x,val_y) = split_trainset(chunked_data, chunked_labels, ratio=split_ratio, batch_size=batch_size)

    # TODO: Note that Robert uses the negative log likelihood loss, which is not provided default
    # by tensorflow.  For now using the cross-entropy.

    num_epochs = 70 # TODO: change to 70, as in Robert's code
    model = make_model(channel, in_len)
    print('Compiling model')
    model.compile(optimizer='adam', loss='mean_squared_logarithmic_error',metrics=['accuracy'])
    print('Fitting data')
    print('training label shape: ', train_y.shape)
    history = model.fit(train_x,train_y,epochs=num_epochs,batch_size=batch_size)
    for idx in range(num_epochs):
        mlflow.log_metric('train_acc',history.history['acc'][idx])
        mlflow.log_metric('train_loss',history.history['loss'][idx])

    print('Evaluating training accuracy')
    train_loss, train_acc = model.evaluate(train_x,train_y)
    print('Evaluating validation accuracy')
    val_loss, val_acc, = model.evaluate(val_x,val_y)

    # Test on all the images.
    for test_img_num in range(len(filenames)):
        (chunked_test_data,test_width,test_height) = chunk_image(merge_dims(train_data[test_img_num]))
        (chunked_test_labels,test_width,test_height) = chunk_image(labels[test_img_num], label=True)
        test_pred_labels = model.predict(chunked_test_data)
        plt.subplot(1,2,1)
        plt.imshow(test_pred_labels.reshape((test_height,test_width)))
        plt.title('Predicted')
        
        plt.subplot(1,2,2)
        plt.imshow(chunked_test_labels.reshape((test_height,test_width)))
        plt.title('Ground Truth')
        test_img_output_name = '../data/out/robert/trained_%s_tested_%s_output.png'%(filenames[img_num],filenames[test_img_num])
        plt.savefig(test_img_output_name)
        mlflow.log_artifact(test_img_output_name)
        test_loss,test_acc = model.evaluate(chunked_test_data,chunked_test_labels)
        mlflow.log_metric('acc_tested_%s'%(filenames[test_img_num],),test_acc)
    #### end for


    mlflow.log_metric('acc_hist',history.history['acc'])
    mlflow.log_metric('loss_hist',history.history['loss'])
    mlflow.log_param('seed_val',seed_val)
    mlflow.log_metric('train_acc', train_acc)
    mlflow.log_metric('val_acc', val_acc)

    print('Train loss: %f, Validation loss: %f' % (train_loss, val_loss))
    print('Train Accuracy: %f Validation accuracy: %f' % (train_acc, val_acc))
    print('Fraction of positives training: %f, validation: %f'%(np.mean(train_y), np.mean(val_y)))

    mlflow.end_run()
    pass

if __name__=='__main__':
    print('Starting program')
    seed_vals = [470127,47012,4701,470,124890436723,1248904367]

#     for s in seed_vals:
    for idx in range(9): # hack, because I know there are nine images
        main(7, 3,img_num=idx,experiment_id='cross_testing')
