import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# print(x_train.shape,"\n",y_train.shape,"\n",x_test.shape,"\n",y_test.shape)
# print()
#print(x_train[10])

plt.matshow(x_train[5])