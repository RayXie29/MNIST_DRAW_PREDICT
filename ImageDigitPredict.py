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

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Arguments
##############################################################
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="Path to source image")
ap.add_argument("-m", "--model_path", required=False, help="Path to trained")
args = vars(ap.parse_args())
##############################################################


def digitPrediction(rois,model):
    rois = np.asarray(rois,dtype = np.uint8).reshape(-1,28,28,1)
    res = model.predict(rois)
    res = np.argmax(res,axis=1)
    return res

def drawRes(img,boxes,res):

    for b,r in zip(boxes,res):
        x,y,w,h = b
        cv2.putText(img,str(r),(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

print('*' * 100)
print('\n\nImage prediction for digit hand writing app\n\n')
print('*' * 100)

print("\n\nLoading the model........\n")
model = load_model(args['model_path'])

dp = DigitParser.DigitParser(threshold = 80)

print("\n\nLoading the image........\n")
dp.SetImgFromFilename(args["image_path"])

print("\n\nProcessing the image........\n")
rois = dp.ParseDigit()

print("\n\nPredict the result........\n")
res = digitPrediction(rois,model)

drawRes(dp.img,dp.boxes,res)

cv2.imshow('prediction_result(Press Esc to close the window)', dp.img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nImage prediction done\n")