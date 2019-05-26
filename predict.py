#import the modules we need for prediction

import pandas as pd
import numpy as np
from keras.models import load_model
import warnings
import argparse
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Arguments
#################################################################################
ap = argparse.ArgumentParser()
ap.add_argument("-i","--test_data",required = True, help = "Path to the testing dataset")
ap.add_argument("-m","--model",required = True, help = "Path to the trained model")
#################################################################################
args = vars(ap.parse_args())

print('*' * 100)
print('\n\nTest data prediction for digit hand writing app\n\n')
print('*' * 100)

print("\n\nLoading the model........\n")
#Load the digit recognizer model
model = load_model(args['model'])
print("\n\nLoading the test data........\n")
test_df = pd.read_csv(args["test_data"])

#process the test dataframe
print("\n\nprocessing the test data........\n")
test_df = test_df/255.0
test_df = test_df.values.reshape(-1,28,28,1)

print("\n\npredicting the data.............\n")

prediction = model.predict(test_df)
prediction = np.argmax(prediction, axis = 1)

output = pd.DataFrame({'ImageId':np.arange(1,prediction.shape[0]+1,1), 'label':prediction})
print("\n\nsaving the result.............\n")
output.to_csv('prediction.csv',index = False)

print("\n\nPredict test data done........\n")