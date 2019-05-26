#import the module we need

import numpy as np # linear algebra
import pandas as pd # data processing

import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


from keras.utils.np_utils import to_categorical #module for one-hot-encodgin
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D #modues for building CNN
from keras.preprocessing.image import ImageDataGenerator #module for data augmentation
from keras.models import Sequential #module for building CNN
from keras.optimizers import adadelta
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

#Arguments
#################################################################################
ap = argparse.ArgumentParser()
ap.add_argument("-i","--train_data", required = True, help = "path to the training dataset")
ap.add_argument("-b","--batch_size", required = False, type = int,  default = 128 ,
                help = "batch size for CNN model training")
ap.add_argument("-e","--epochs", required = False,type = int, default = 30,
                help = "epochs for CNN model training")

ap.add_argument("-v","--evaluation_flag", required = False, type = bool, default= False,
                help = "Flag for running the model evaluation(including train_test_split, confusion Matrix")

ap.add_argument("-m","--model_path", required = False , default = None,
                help = "Path to the model you want to train")

ap.add_argument("-s","--show_flag", required = False, type = bool, default = True,
                help = "Flag for showing the data, training... detail information or not")

args = vars(ap.parse_args())
#################################################################################

print('*' * 100)
print('\n\nModel training for digit hand writing app\n\n')
print('*' * 100)

show_flag = args["show_flag"]


#Data Loading & Checking
data_dir = args["train_data"]

#Check the shape of dataset first
train_df = pd.read_csv(data_dir)
print('Shape of training data : ', train_df.shape)

if show_flag == True:
    # Check out some detail of training dataset
    print("........First 5 rows of data........")
    print(train_df.head(5))
    print("....................................")


train_label = train_df.label
train_df = train_df.drop(['label'], axis = 1)

if show_flag == True:
    # check the counts in label
    print('\n\nTraining labels bar plot...........')
    plt.figure(figsize=(8, 6))
    plt.bar(np.arange(0, 10, 1), train_label.value_counts().sort_index().values)
    plt.xlabel('labels')
    plt.ylabel('count')
    plt.title('label counts in training dataset')
    plt.show()

#checking the missing values
print("\n.......Check the missing values in training dataset.......")
nans = np.sum(train_df.isnull().sum())
print('Missing values in training dataset : %d' %nans)

if nans != 0:
    k = input('Do you want to drop all the missing rows? y/n : ')
    if k == 'y':
        idx = pd.isnull(train_df).any(1).nonzero()[0]
        train_df = train_df.drop(idx,axis=0)
        print('Shape of training data after drop the missing rows :',train_df.shape)

print("..........................................................\n")

if show_flag == True:
    # checking the data type of pixels
    print("\n.......Check the data describe.......")
    print(train_df.iloc[0].describe())
    print(".......................................\n")

#data preparation
#data normalization
#The pixel values in training & testing dataset are only 0 & 255
#So we could normalize the pixel values into 0 to 1, just like binary type
train_df = train_df/255.0

#reshape image
#Inorder to represent the pixel values in picture, we could reshape the data into square shape (784,1) -> (28,28,1).
#The 1 means the images are grayscale
train_df = train_df.values.reshape(-1,28,28,1)

if show_flag == True:
    print("\ncheck the picture of data...........\n\n")
    plt.imshow(train_df[3][:, :, 0], cmap=plt.cm.binary)
    plt.title('label : %d' % train_label[3])
    plt.show()

#one hot encoding the training labels
#There will be 10 classes in the label ( 0 ~ 9)
train_y = to_categorical(train_label,num_classes = 10)

#Split the training dataset into train & validation dataset for model training & checking
x_train, x_val, y_train, y_val = train_test_split(train_df,train_y,test_size= 0.2,random_state=2)


#Data Augmentation
#rotation_range will rotate the image
#width_shift_range will shift the image horizontally
#height_shidt_range will shift the image vertically
#zoom_range will zoom the image
dataGenerator = ImageDataGenerator( rotation_range = 10,
                                    width_shift_range = 0.1,
                                    height_shift_range = 0.1,
                                    zoom_range = 0.1 )

dataGenerator.fit(x_train)


print('\n\nModel building........\n')

def build_model():
    # Build CNN Model
    # Use keras Sequential module to build our CNN model, it will stack the layer by adding one  layer a time.
    model = Sequential()
    # Conv2D will use the filter to extract the information from original image(2Darray)
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    # MaxPooling2D is used for downsampling the data. It can reduce the estimation effort and overfitting.
    model.add(MaxPool2D(pool_size=(2, 2)))
    # Dropout is set to ignore the nodes randomly. It will drop a propotion of the NN and improve the overfitting problem.
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    # Flatten will turn the final feature into 1D array(vector)
    model.add(Flatten())
    # Dense is used for convergence the result, which is more like a classifier.
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss="categorical_crossentropy", optimizer= adadelta(), metrics=['accuracy'])
    return model


model = None
if args['model_path'] == None:
    model = build_model()
else :
    model = load_model(args['model_path'])

batch_size = args["batch_size"]
epochs = args["epochs"]

if args["evaluation_flag"] == True:

    print('\nEnter model evaluation section.......')
    print('Training for evaluation.......\n\n')
    history = model.fit_generator(dataGenerator.flow(x_train, y_train, batch_size),
                                  epochs=epochs, verbose=1, validation_data=(x_val, y_val),
                                  steps_per_epoch=x_train.shape[0] / batch_size)

    score = model.evaluate(x_val, y_val, verbose=0)

    print('\n\n Result:\n')

    print("Total loss: {}", format(score[0]))
    print("Total accuracy: {}", format(score[1]))

    if show_flag == True:
        print('\n\nAccuracy & Loss plot\n')

        plt.figure(figsize=(10, 12))

        plt.subplot(2, 1, 1)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['training accuracy', 'validation accuracy'])
        plt.title('Accuracy v.s. Epoch')

        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['training loss', 'validation loss'])
        plt.title('Loss v.s. Epoch')
        plt.show()

        print('\n\nConfusion Matrix plot\n')
        y_prediction = model.predict(x_val)
        # np argmax will turn the one hot encoding vector into just one number which is our label
        y_prediction = np.argmax(y_prediction, axis=1)
        y_true = np.argmax(y_val, axis=1)
        confusionMat = confusion_matrix(y_true, y_prediction)

        plt.figure(figsize=(12, 10))
        sns.heatmap(confusionMat, cmap=plt.cm.summer, annot=True)
        plt.title('Confusion Matrix of CNN Model')
        plt.xlabel('Prediction label')
        plt.ylabel('True label')
        plt.show()



print('\n\nModel training........\n')

model.fit_generator(dataGenerator.flow(train_df,train_y,batch_size = batch_size),
                    epochs = epochs, verbose = 2, steps_per_epoch = train_df.shape[0]/batch_size)


print('\n\nSaving the model....\n')
model.save('mnist_model.h5')
print('\n\nModel training done....\n')

