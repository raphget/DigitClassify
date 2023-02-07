
NB_CLASSES=10
import numpy as np
from keras.utils import np_utils
from sklearn import preprocessing

def processTestData(X, y):

    # X preprocessing goes here -- students optionally complete
    X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    # y preprocessing goes here.  y_test becomes a ohe
    y_ohe = np_utils.to_categorical (y, NB_CLASSES)
    return X, y_ohe