#Import the modules we need

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import argparse
import cv2
import warnings
import numpy as np
import copy
import sys
warnings.filterwarnings('ignore')


#Arguments
#################################################################################
ap = argparse.ArgumentParser()
ap.add_argument("-m","--model_path",required = True, help = "Path to the model you want to train")
ap.add_argument("-b","--batch_size",required = False, type = int, default= 1,
                help = "Batch size for model training")
ap.add_argument("-e","--epochs", required = False, type = int, default = 50,
                help = 'Epochs for model training')
args = vars(ap.parse_args())
#################################################################################

def DrawMouseEvent(event, x, y, flags, param):

    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
       param[0] = x,y

def DigitDrawTrain(images):

    img = np.zeros((256,256,1),np.uint8)

    center = [(-1,-1)]

    cv2.namedWindow("Digit Drawing(Press q to stop the drawing)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Digit Drawing(Press q to stop the drawing)",DrawMouseEvent,center)
    count = 0
    while(1):
        cv2.imshow("Digit Drawing(Press q to stop the drawing)", img)
        k = cv2.waitKey(1)

        if center[0][0] != -1 and center[0][1] != -1:
            cv2.circle(img,(center[0][0],center[0][1]),15,(255,255,255),-1)

        if k == ord('p'):
            train_image = copy.deepcopy(img)
            train_image = cv2.resize(train_image,(28,28),cv2.INTER_NEAREST)
            train_image = train_image / 255
            train_image = train_image.reshape(28,28,1)
            images.append(train_image)
            center[0] = -1, -1
            img = np.zeros((256, 256, 1), np.uint8)
            print(f'Digit {count} finished')
            count += 1

        elif k == ord('r'):
            center[0] = -1, -1
            img = np.zeros((256, 256, 1), np.uint8) * 255

        elif k == ord('q'):
            break

        if count == 10:
            break

    cv2.destroyAllWindows()

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
print("\n\nDraw for model training! press p for data submit, r for re-drawing the picture and q for end the program")
print("Please draw the number 0 - 9 (ascending)\n\n")
print('*' * 100)

images = []
DigitDrawTrain(images)
cv2.destroyAllWindows()
if len(images) == 0:
    sys.exit('\n\nNo Input Image, exit......')


print("\n\nProcessing the data........\n")


kernel = np.ones((3,3),np.uint8)
labels = np.arange(0,len(images),1)
labels = expandingData(images,labels)
images = np.asarray(images).reshape(-1,28,28,1)

print("\n\nLoading the model........\n")
model = load_model(args["model_path"])
dataGenerator = ImageDataGenerator( rotation_range = 10,
                                    width_shift_range = 0.1,
                                    height_shift_range = 0.1,
                                    zoom_range = 0.1 )

dataGenerator.fit(images)
print('\n\nModel training........\n')

model.fit_generator(dataGenerator.flow(images,labels,batch_size = args['batch_size']),
                    epochs = args['epochs'],verbose = 2,
                    steps_per_epoch = images.shape[0]/args['batch_size'])


print('\n\nSaving the model....\n')
model.save('mnist_model.h5')
print("\nDraw Training Done........\n")
