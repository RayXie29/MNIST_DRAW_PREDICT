#import the modules we need

import cv2
from keras.models import load_model
import numpy as np
import os
import warnings
import argparse
import copy

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def DrawMouseEvent(event, x, y, flags, param):

    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
       param[0] = x,y

def DigitDrawPredict(model):

    img = np.zeros((256,256,1),np.uint8)

    center = [(-1,-1)]

    cv2.namedWindow("Digit Drawing(Press q to close the program)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Digit Drawing(Press q to close the program)",DrawMouseEvent,center)

    while(1):
        cv2.imshow("Digit Drawing(Press q to close the program)", img)
        k = cv2.waitKey(1)

        if center[0][0] != -1 and center[0][1] != -1:
            cv2.circle(img,(center[0][0],center[0][1]),15,(255,255,255),-1)

        if k == ord('p'):
            predict_img = copy.deepcopy(img)
            predict_img = cv2.resize(img, (28, 28), cv2.INTER_NEAREST)

            predict_img = predict_img.reshape(1, 28, 28, 1)

            prediction = model.predict(predict_img)
            prediction = np.argmax(prediction, axis=1)

            print('prediction result : %d ' %prediction)

            center[0] = -1, -1
            img = np.zeros((256, 256, 1), np.uint8)
        elif k == ord('r'):
            center[0] = -1, -1
            img = np.zeros((256, 256, 1), np.uint8)
        elif k == ord('q'):
            break


#Arguments
#################################################################################
ap = argparse.ArgumentParser()
ap.add_argument("-i","--model",required = True, help = "Path to the trained model")
args = vars(ap.parse_args())
#################################################################################

print('*' * 100)
print('\n\nDrawing prediction for digit hand writing app\n\n')
print('*' * 100)

print("\n\nLoading the model........\n")
model = load_model(args["model"])
print('*' * 100)
print("\n\nDraw for prediction! press p for predict, r for re-drawing the picture and q for end the program\n\n")
print('*' * 100)
DigitDrawPredict(model)

print("Drawing prediction done")

