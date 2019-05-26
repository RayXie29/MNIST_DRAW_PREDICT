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


def retrieveContours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blur, 75, 180)
    canny = cv2.morphologyEx(canny,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8) )
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def retrieveBoundingBoxes(contours):
    boundingBoxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 500 and w/h < 5 and h/w < 5:
            boundingBoxes.append(np.array([x, y, w, h], dtype=int))

    boundingBoxes = np.asarray(boundingBoxes, dtype=int)
    return boundingBoxes

def croppingBoxes(img,boxes):
    rois = []
    for b in boxes:
        x,y,w,h = b
        roi = img[y:y+h,x:x+w]
        rois.append(roi)

    return rois

def img_preprocessing(rois,extend):

    for i,roi in enumerate(rois):
        roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        roi = cv2.copyMakeBorder(roi,extend,extend,extend,extend,cv2.BORDER_CONSTANT,value = (255,255,255) )
        cv2.imshow('roi',roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        ret, roi = cv2.threshold(roi,120,255,cv2.THRESH_BINARY_INV)
        roi = cv2.resize(roi,(28,28),cv2.INTER_NEAREST)


        rois[i] = roi.reshape(28,28,1)

    return rois

def digitPrediction(rois,model):
    rois = np.asarray(rois,dtype = np.uint8).reshape(-1,28,28,1)
    res = model.predict(rois)
    res = np.argmax(res,axis=1)
    return res

def drawRes(img,boxes,res):

    for b,r in zip(boxes,res):
        x,y,w,h = b
        cv2.putText(img,str(r),(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)


print('*' * 100)
print('\n\nImage prediction for digit hand writing app\n\n')
print('*' * 100)

print("\n\nLoading the model........\n")
model = load_model(args['model_path'])

print("\n\nLoading the image........\n")
img = cv2.imread(args["image_path"], cv2.IMREAD_COLOR)
img = cv2.resize(img, (640, 480))

print("\n\nProcessing the image........\n")
contours = retrieveContours(img)
boxes = retrieveBoundingBoxes(contours)

rois = croppingBoxes(img,boxes)
rois = img_preprocessing(rois,20)

print("\n\nPredict the result........\n")
res = digitPrediction(rois,model)

drawRes(img,boxes,res)

cv2.imshow('prediction_result(Press Esc to close the window)', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nImage prediction done\n")