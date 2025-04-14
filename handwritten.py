import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test)=tf.keras.datasets.mnist.load_data()

x_train = x_train/255.0
x_test = x_test/255.0
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)
