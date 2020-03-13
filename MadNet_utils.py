from keras.backend import cast_to_floatx
from keras.datasets import cifar10,mnist
import numpy as np

def normalize_05(data):
    return (data / 255.0) - 0.5


def load_cifar10_05():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = normalize_05(cast_to_floatx(X_train))
    X_test = normalize_05(cast_to_floatx(X_test))
    return (X_train, y_train), (X_test, y_test)

def load_mnist_05():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.expand_dims(X_train,-1)
    X_test = np.expand_dims(X_test,-1)
    X_train = normalize_05(cast_to_floatx(X_train))
    X_test = normalize_05(cast_to_floatx(X_test))
    return (X_train, y_train.astype(np.int)), (X_test, y_test.astype(np.int))