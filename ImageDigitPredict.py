# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:39:59 2019

@author: Ray_Xie
"""

import cv2
import argparse
import warnings
import numpy as np
from keras.models import load_model

#Arguments
##############################################################
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image_path",required = True, help = "Path to source image")
ap.add_argument("-m","--model_path",required = False, help = "Path to trained")
args = vars(ap.parse_args())
##############################################################


def retrieveContours(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),0)
    canny = cv2.Canny(blur,75,180)
    contours,hierachy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def retrieveBoundingBoxes(contours):
    boundingBoxes = []

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h > 500:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            boundingBoxes.append(np.array([x,y,w,h],dtype = int))
        
    boundingBoxes = np.asarray(boundingBoxes,dtype = int)
    return boundingBoxes

model= load_model(args['model_path'])
img = cv2.imread(args["image_path"],cv2.IMREAD_COLOR)
img = cv2.resize(img,(640,480))
contours = retrieveContours(img)
Boxes = retrieveBoundingBoxes(contours)




#cv2.drawContours(img,contours,-1,(0,255,0),2)

cv2.imshow('contours',img)

cv2.waitKey(0)
cv2.destroyAllWindows()