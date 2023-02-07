# purpose: Keras model(Dropout Regularization)
# -------------------------------------------------

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from tensorflow import math
from keras import layers
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

from preprocess import processTestData
import argparse

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from confusionMatrixHeatmap import confusionMatrixHeatmap
from barPlotTemplate import generateBarPlot

def parseArguments():
    parser = argparse.ArgumentParser(
        description='Build a Keras model for Image classification')

    parser.add_argument('--training_x', action='store',
                        dest='XFile', default="", required=True,
                        help='matrix of training images in npy')
    parser.add_argument('--training_y', action='store',
                        dest='yFile', default="", required=True,
                        help='labels for training set')

    parser.add_argument('--outModelFile', action='store',
                        dest='outModelFile', default="", required=True,
                        help='model name for your Keras model')

    return parser.parse_args()

# Builds heatmap based on model confusion matrix
def generateHeatMap(y_test,y_pred, outfile):
    heat_matrix = math.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)).numpy()
    # Normalize heatmap values
    heat_matrix = heat_matrix.astype('float') / heat_matrix.sum(axis=1)[:, np.newaxis]
    confusionMatrixHeatmap(heat_matrix, outfile)

# Initializes and configures sequential network (feedforward)
def generateModel(X_train,y_train,X_test,y_test):
    # convert the labels from integers to vectors
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    #Build sequential model (feedforward)
    model = Sequential()
    # Add layers:
    #   Input(784) -> h1(500) -> h2(500) -> Output(10)
    model.add(layers.Input(shape=(784,)))
    model.add(Dense(500,  activation="sigmoid"))
    model.add(layers.Dropout(0.5))
    model.add(Dense(500, activation="sigmoid"))
    model.add(layers.Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    # Configure model for training
    model.compile(loss="categorical_crossentropy", optimizer= SGD(0.1), metrics=["accuracy"])
    # Trains the model: epochs=250, batch_size=100
    model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=250, batch_size=256)
    return model

def main():
    np.random.seed(1671)
    parms = parseArguments()

    #load and preprocess training files
    X_train = np.load(parms.XFile)
    y_train = np.load(parms.yFile)
    (X_train, y_train) = processTestData(X_train,y_train)

    #load and preprocess test files
    X_test = np.load('./resources/MNIST_X_test_1.npy' ) 
    y_test = np.load('./resources/MNIST_y_test_1.npy')
    (X_test, y_test) = processTestData(X_test,y_test)
    
    print('KERAS modeling build starting...')

    model = generateModel(X_train,y_train,X_test,y_test)

    ## save your model
    model.save(parms.outModelFile)
    
    # Store predicted values for test/training data in a 1d array
    y_pred = model.predict(X_test)
    generateHeatMap(y_test,y_pred,"./output/hmm3.pdf")

    # Generate a text report showing the main classification metrics.
    print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), zero_division = 0))

    # Generate report as Dictionary
    report = classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), zero_division = 0, output_dict=True)

    generateBarPlot(report, 'M3 Performance',"./output/barChartm3.pdf")

    ## save your model
    #model.save(parms.outModelFile)


if __name__ == '__main__':
    main()
    
