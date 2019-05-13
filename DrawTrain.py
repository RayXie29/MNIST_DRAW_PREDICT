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

def DrawMouseEvent(event, x, y, flags, param):

    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
       param[0] = x,y

def DigitDrawTrain(images):

    img = np.zeros((256,256,1),np.uint8)

    center = [(-1,-1)]

    cv2.namedWindow("Digit Drawing(Press q to stop the drawing)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Digit Drawing(Press q to stop the drawing)",DrawMouseEvent,center)

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

        elif k == ord('r'):
            center[0] = -1, -1
            img = np.zeros((256, 256, 1), np.uint8) * 255

        elif k == ord('q'):
            break

        if count == 10:
            break

    cv2.destroyAllWindows()


#Arguments
#################################################################################
ap = argparse.ArgumentParser()
ap.add_argument("-i","--model_path",required = True, help = "Path to the model you want to train")
ap.add_argument("-b","--batch_size",required = False, type = int, default= 1,
                help = "Batch size for model training")
ap.add_argument("-e","--epochs", required = False, type = int, default = 50,
                help = 'Epochs for model training')
args = vars(ap.parse_args())
#################################################################################

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

length = len(images)
for i in range(0,length):
    erosion = cv2.erode(images[i],kernel,iterations=1)
    erosion = erosion.reshape(28,28,1)
    images.append(erosion)

for i in range(0,length):
    dilation = cv2.dilate(images[i], kernel, iterations=1)
    dilation = dilation.reshape(28,28,1)
    images.append(dilation)


label = np.arange(0,length,1)
labels = np.concatenate((label,label))
labels = np.concatenate((labels,label))
labels = to_categorical(labels,10)
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
