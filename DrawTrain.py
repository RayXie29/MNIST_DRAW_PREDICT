#Import the modules we need

from keras.models import load_model
import argparse
import cv2
import warnings
import numpy as np
import copy

warnings.filterwarnings('ignore')

#Arguments
#################################################################################
ap = argparse.ArgumentParser()
ap.add_argument("-i","--model_path",required = True, help = "Path to the model you want to train")
args = vars(ap.parse_args())
#################################################################################


def DrawMouseEvent(event, x, y, flags, param):

    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
       param[0] = x,y
'''
def DigitDrawPredict(model):

    img = np.ones((256,256,1),np.uint8) * 255

    center = [(-1,-1)]

    cv2.namedWindow("Digit Drawing(Press Ecs to close the program)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Digit Drawing(Press Ecs to close the program)",DrawMouseEvent,center)

    while(1):
        cv2.imshow("Digit Drawing(Press Ecs to close the program)", img)
        k = cv2.waitKey(1)

        if center[0][0] != -1 and center[0][1] != -1:
            cv2.circle(img,(center[0][0],center[0][1]),15,(0,0,0),-1)

        if k == ord('p'):
            predict_img = copy.deepcopy(img)
            predict_img = cv2.resize(img, (28, 28), cv2.INTER_NEAREST)

            cv2.imshow('predict_img',predict_img)
            cv2.waitKey(0)
            cv2.destroyWindow('predict_img')

            predict_img = predict_img / 255.0
            predict_img = predict_img.reshape(1, 28, 28, 1)

            prediction = model.predict(predict_img)
            prediction = np.argmax(prediction, axis=1)

            print(prediction)

            center[0] = -1, -1
            img = np.ones((256, 256, 1), np.uint8) * 255

        elif k == ord('q'):
            break
'''