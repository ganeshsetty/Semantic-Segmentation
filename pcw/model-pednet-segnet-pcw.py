#### Keras updated from 1.2.0 to 2.0.2 , resulted in Conv2D etc

from __future__ import absolute_import
from __future__ import print_function
import os

import keras

#import keras.models as models
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D,Conv2D
from keras.layers.normalization import BatchNormalization


from keras import backend as K
K.set_image_dim_ordering('th')

import cv2
import numpy as np
import json
np.random.seed(7) # 0bserver07 for reproducibility

data_shape = 360*480

#common net
main_input = Input(shape=(3,360,480), dtype='float32', name='main_input')
Conv2D_1 = Conv2D(96,(11,11), strides=(4,4), activation='relu', padding='valid')(main_input)
MaxP2D_1 = MaxPooling2D(pool_size=(3, 3),strides=(2,2))(Conv2D_1)
Conv2D_2 = Conv2D(256,(5,5), strides=(1,1), activation='relu', padding='valid')(MaxP2D_1)
MaxP2D_2 = MaxPooling2D(pool_size=(3, 3),strides=(2,2))(Conv2D_2)

Flatten_1 = Flatten()(MaxP2D_2)

#pedestrian specific net
Conv2D_3 = Conv2D(384,(3,3), strides=(1,1), activation='relu', padding='valid')(MaxP2D_2)
Conv2D_4 = Conv2D(256,(3,3), strides=(1,1), activation='relu', padding='valid')(Conv2D_3)
FC1_1 = Dense(256, activation = 'relu')(Flatten_1)

# Semantic segnet specific
#FC3_1 = Dense(2048, activation = 'relu')(Flatten_1)
FC3_1 = Dense(256, activation = 'relu')(Flatten_1)
segnet_output = Dense(172800, activation='sigmoid', name='segnet_output')(FC3_1)

# merge from FC3 o/p to FC1 o/p as input to FC2
merged = keras.layers.concatenate([FC1_1, FC3_1])
FC2_1 = Dense(256, activation = 'relu')(merged)
pednet_output = Dense(2, activation = 'softmax', name='pednet_output')(FC2_1)

pcw_net = Model(main_input, outputs=[pednet_output, segnet_output])

# Save model to JSON

with open('pcw_model.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(pcw_net.to_json()), indent=2))
