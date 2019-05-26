# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:39:59 2019

@author: Ray_Xie
"""

import cv2
import numpy as np

class DigitParser:

    __SetFlag = False

    def __init__(self, extending = 20, resizing = (640,480), threshold = 100, Cth1 = 75, Cth2 = 180, bsize = 500):
        # Extend length for copyMakeBorder
        self.extend = extending
        # Size for resizing the input image
        self.size = resizing
        # Threshold for thresholding the image to retrieve digits
        self.th = threshold
        # Threshold for canny function
        self.Cth1 = Cth1
        self.Cth2 = Cth2

        # Threshold size for roi bounding box
        self.bsize = bsize

    def SetImage(self,img):
        self.img = cv2.resize(img,self.size)
        self.__SetFlag = True

    def SetImgFromFilename(self,filename):
        self.img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        self.img = cv2.resize(self.img,self.size)
        self.__SetFlag = True


    def __retrieveContours(self):

        gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        canny = cv2.Canny(blur, self.Cth1, self.Cth2)
        canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.contours = contours

        return contours

    def __retrieveBoundingBoxes(self,contours):

        boundingBoxes = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > self.bsize and w / h < 5 and h / w < 5:
                boundingBoxes.append(np.array([x, y, w, h], dtype=int))

        boundingBoxes = np.asarray(boundingBoxes, dtype=int)

        self.boxes = boundingBoxes

        return boundingBoxes

    def __croppingBoxes(self,boxes):


        rois = []
        for b in boxes:
            x, y, w, h = b
            roi = self.img[y:y + h, x:x + w]
            rois.append(roi)

        return rois

    def __imgPreProcessing(self,rois):

        for i, roi in enumerate(rois):
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.copyMakeBorder(roi, self.extend, self.extend, self.extend, self.extend, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            ret, roi = cv2.threshold(roi, self.th, 255, cv2.THRESH_BINARY_INV)
            roi = cv2.resize(roi, (28, 28), cv2.INTER_NEAREST)

            rois[i] = roi.reshape(28, 28, 1)

        return rois

    def ParseDigit(self):

        if self.__SetFlag == False:
            print("Please set the image for DigitParser")
            return -1

        contours = self.__retrieveContours()
        boxes = self.__retrieveBoundingBoxes(contours)
        rois = self.__croppingBoxes(boxes)
        rois = self.__imgPreProcessing(rois)

        if len(rois) == 0:
            print("Parsing Digit fail")
            return -1
        else:
            return rois

