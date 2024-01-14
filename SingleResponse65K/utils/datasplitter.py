
from joblib import load
from utils.classes import SpectralDataset
from sklearn.model_selection import train_test_split
import numpy as np


def splitter(RANDOM_STATE=42, TEST_SIZE=.25, skip_normalization=False):

    X, y = load('/home/djh/Big8_testing/data/big8/X.joblib'), load('/home/djh/Big8_testing/data/big8/Y.joblib')
    print(X.shape[0], '<- shape 0 of dataset')
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE)

    cmap_list = []
    amap_list = []
    print(X_val)
    for i in X_val['2200.0']:
        if i > 2:
            cmap_list.append('red')
        else:
            cmap_list.append('blue')

    amap = np.array(amap_list)
    cmap = np.array(cmap_list)

    train_dataset = SpectralDataset(X_train, y_train, skip_normalization=skip_normalization)
    test_dataset = SpectralDataset(X_test, y_test, skip_normalization=skip_normalization)
    val_dataset = SpectralDataset(X_val, y_val, skip_normalization=skip_normalization)
    return cmap, train_dataset, test_dataset, val_dataset

cmap, train_dataset, test_dataset, val_dataset = splitter()