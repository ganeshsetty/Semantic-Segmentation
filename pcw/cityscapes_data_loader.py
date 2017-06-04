from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np
import itertools

from helper import *
import os

import keras
from keras.utils import np_utils

from keras import backend as K
K.set_image_dim_ordering('th')

# Copy the cityscapes 
DataPath = './CityScapes/'
data_shape = 1024*2048


def load_data(mode):
    data = []
    label = []
    label_pedestrian = []
	
    with open(DataPath + mode +'.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
    for i in range(len(txt)):
        data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + txt[i][0][7:])),2))
        #label.append(one_hot_it(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[:,:,0]))
        label.append(cv2.imread(os.getcwd() + txt[i][1][7:],0))
        #label_pedestrian.append(txt[i][2])
        label_pedestrian.append(np_utils.to_categorical(txt[i][2],2))
        print('.',end='')
    return np.array(data),np.array(label), np.array(label_pedestrian)



train_data, train_label,train_label_pedestrian = load_data("train_pedestrian_aug_new")
train_label = np.reshape(train_label,(510,172800))
train_label_pedestrian = np.reshape(train_label_pedestrian,(510,2))

test_data, test_label,test_label_pedestrian = load_data("test_pedestrian_org")
test_label = np.reshape(test_label,(233,172800))
test_label_pedestrian = np.reshape(test_label_pedestrian,(233,2))

np.save("data/train_data", train_data)
np.save("data/train_label", train_label)
np.save("data/train_label_pedestrian", train_label_pedestrian)

np.save("data/test_data", test_data)
np.save("data/test_label", test_label)
np.save("data/test_label_pedestrian", test_label_pedestrian)

label_array = np.load('data/train_label_pedestrian.npy')
label_seg_array = np.load('data/train_label.npy')
print(label_array[0])
print(label_array[3])
print(label_array[5])
print(label_array[9])

print(label_seg_array.shape)
print(label_seg_array[100].shape)
print(label_seg_array[200])
#np.save("data/val_data", val_data)
#np.save("data/val_label", val_label)

# FYI they are:
# Sky = [128,128,128]
# Building = [128,0,0]
# Pole = [192,192,128]
# Road_marking = [255,69,0]
# Road = [128,64,128]
# Pavement = [60,40,222]
# Tree = [128,128,0]
# SignSymbol = [192,128,128]
# Fence = [64,64,128]
# Car = [64,0,128]
# Pedestrian = [64,64,0]
# Bicyclist = [0,128,192]
# Unlabelled = [0,0,0]
