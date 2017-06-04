from __future__ import absolute_import
from __future__ import print_function
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=None'


from keras import optimizers
import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D,Conv2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

from keras import backend as K
K.set_image_dim_ordering('th')

import cv2
import numpy as np
import json
np.random.seed(7) # 0bserver07 for reproducibility


data_shape = 360*480

#class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]

#class_weighting = [1.5,1007]

def mean_euc_dist_sq(y_true, y_output):
    return (K.mean(K.sum(K.square(y_true - y_output), axis=-1)))/2.0

# load the data
train_data = np.load('./data/train_data.npy')
train_label = np.load('./data/train_label.npy')
train_label_pedestrian = np.load('./data/train_label_pedestrian.npy')

test_data = np.load('./data/test_data.npy')
test_label = np.load('./data/test_label.npy')
test_label_pedestrian = np.load('./data/test_label_pedestrian.npy')

#print("train_label_pedestrian:",train_label_pedestrian[9])

# load the model:
with open('pcw_model.json') as model_file:
    pcw_net = models.model_from_json(model_file.read())


sgd = optimizers.SGD(lr=0.001,decay=1e-4,momentum=1.9)
#pcw_net.compile(loss = 'categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


pcw_net.compile(optimizer='sgd',
              loss={'pednet_output': 'categorical_crossentropy', 'segnet_output': mean_euc_dist_sq},
              loss_weights={'pednet_output': 1., 'segnet_output': 0.01},metrics=['accuracy'])

#pcw_net.compile(loss = 'categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# checkpoint
filepath="weights.pcwnet.pcw.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

nb_epoch = 50
#batch_size = 10
batch_size = 128

# Fit the model
#ped,seg =pcw_net.fit(train_data, train_label_pedestrian, callbacks=callbacks_list, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(test_data, test_label_pedestrian), shuffle=True) # validation_split=0.33

history = pcw_net.fit({'main_input':train_data},{'pednet_output':train_label_pedestrian, 'segnet_output':train_label},epochs = 100,batch_size =batch_size,shuffle=True)

# This save the trained model weights to this file with number of epochs
pcw_net.save_weights('model_weight_pcwnet{}.hdf5'.format(nb_epoch))

