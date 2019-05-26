# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:39:59 2019

@author: Ray_Xie
"""

import cv2
import argparse
import warnings
import numpy as np
import os
import DigitParser
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Arguments
##############################################################
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="Path to source image")
ap.add_argument("-m", "--model_path", required=False, help="Path to trained model")
ap.add_argument("-b","--batch_size",required = False, type = int, default= 1,
                help = "Batch size for model training")
ap.add_argument("-e","--epochs", required = False, type = int, default = 50,
                help = 'Epochs for model training')
args = vars(ap.parse_args())
##############################################################


def digitLabeling(rois):
    labels = []
    for roi in rois:
        temp = roi.reshape(28,28,1)
        temp = cv2.resize(temp,(256,256))

        cv2.imshow('roi(Label by press the digit number on keyboard)', temp)
        k = int(cv2.waitKey(0))
        labels.append(k-48)

        cv2.destroyWindow('roi(Label by press the digit number on keyboard)')

    return labels

def expandingData(rois,labels):
    extend_labels = np.concatenate((labels, labels))
    extend_labels = to_categorical(extend_labels, 10)

    kernel = np.ones((3, 3), np.uint8)
    rois_len = len(rois)

    for i in range(0, rois_len):
        dilation = cv2.dilate(rois[i], kernel, iterations=1)
        dilation = dilation.reshape(28, 28, 1)
        rois.append(dilation)

    return extend_labels



print('*' * 100)
print('\n\nTraining by image digits\n\n')
print('*' * 100)

print("\n\nLoading the model........\n")
model = load_model(args['model_path'])

dp = DigitParser.DigitParser(threshold = 80)

print("\n\nLoading the image........\n")
dp.SetImgFromFilename(args["image_path"])

print("\n\nProcessing the image........\n")
rois = dp.ParseDigit()

print("\n\nLabel the digits........\n")
labels = digitLabeling(rois)

labels = expandingData(rois,labels)

rois = np.asarray(rois,dtype = np.uint8).reshape(-1,28,28,1)
dataGenerator = ImageDataGenerator(rotation_range = 10,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                    zoom_range = 0.1 )

dataGenerator.fit(rois)

print('\n\nModel training........\n')

model.fit_generator(dataGenerator.flow(rois,labels,batch_size = args["batch_size"]),
                    epochs = args["epochs"], verbose = 2, steps_per_epoch = len(rois)/args["batch_size"])

print('\n\nSaving the model....\n')
model.save('mnist_model.h5')

print("\nImage digit training done\n")