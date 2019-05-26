# CNN Mnist/Draw/Image Train/Predict
<br />
This repo is using CNN model to train a digit recognizer for some interesting applcation. The Whole model is built by Keras sequential module. <br />

## Train folder
Contain several training python file, including model training by dataset <br /><br />
1.Csv file: This training dataset is the same format as the digit recognizer competition in Kaggle. The first column would be the training label and the rest of the column would be the pixel values of digit image. <br /><br />
2.Image: This format of data will received by using the ImageDigitTrain.py. It will use DigitParser class to parse the digit image in the input image. <br /><br />
3.Draw: In DrawTrain.py, it is implement by openCV module. The input images are drawn by the user and the digit data will in an ascending order.<br /><br />
<br />
**train.py** : <br /><br />
Train the module by csv file. It can show some data visualized information(label counts, confusion matrix, accuracy/loss curve....) by setting the show_flag to true. Only the bacth_size and the epochs are tunable, if further adjustment wants to do, then you can modify the source code for this purpose. After training the model, it will save a model file. Also you can import the trained model for further training by setting the model_path argument<br /><br />
![alt text](https://raw.githubusercontent.com/RayXie29/MNIST_DRAW_PREDICT/master/imgs/csv_train.png)<br />
<br /><br />
**DrawTrain.py** : <br /><br />
This python file will train the model by your own drawing picture. You need to draw 10 digits in an ascending order(0~9), and the model will trained by the digits image you draw. But this might has very little effect, since the training data is too less, but it will use cv2.dilate to make the digits thicker and ImageDataGenerator for expanding the dataset.<br /><br />
![alt text](https://raw.githubusercontent.com/RayXie29/MNIST_DRAW_PREDICT/master/imgs/drawTrain.gif)<br />
<br /><br />
**ImageDigitTrain.py** : <br /><br />
This python file will take an image for input and use DigitParser class to parse the digits in the image, and you need to label the digits by yourself. The DigitParser class is written by several openCV functions, but it might only can handle some simple situation(Normal size digits in a simple backgound).<br /><br />
![alt text](https://raw.githubusercontent.com/RayXie29/MNIST_DRAW_PREDICT/master/imgs/ImageDigitTrain.gif)<br />
<br />
<br />
## Predict folder
Contain severl python files which are for predicting the result. <br />
<br />
**predict.py** : <br />
This python file is for csv file data prediction. Simple input the model path adn the test data, it will output a prediction.csv for result. The test data should have the same format as Kaggle`s(784 columns, eacj column represent pixel value)<br /><br />
**predictDraw.py** : <br />
This python file will predict the digit result you draw. By the way, the canvas is maded by openCV function, every time you press the left mouse button, it will draw an small circle on the black image. When there are many small circle connect togetherm it will look like a line.<br /><br />
![alt text](https://raw.githubusercontent.com/RayXie29/MNIST_DRAW_PREDICT/master/imgs/predictDraw.gif)<br />
<br /><br />
**ImageDigitPredict.py** : <br />
This python file is very similar to ImageDigitTrain.py. It will take an image and model for input, then it will use DigitParser to parse the digit images in the input image. Then use the model to predict the digit images and show the result on the original input image.<br /><br />
![alt text](https://raw.githubusercontent.com/RayXie29/MNIST_DRAW_PREDICT/master/imgs/ImageDigitPredict.png)<br />
<br />
<br />
## extra_modules folder
Some modules for image processing
