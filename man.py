import tensorflow as tf
from tensorflow.keras import datasets,layers,models
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# print(x_train.shape,"\n",y_train.shape,"\n",x_test.shape,"\n",y_test.shape)
# print()
#print(x_train[10])

x_train = x_train/255
x_test = x_test/255

print(x_train.shape,x_test.shape)

x_train_flat = x_train.reshape(len(x_train), 28*28)
x_test_flat = x_test.reshape(len(x_test), 28*28)
print(x_train_flat.shape,x_test_flat.shape)
print(x_train_flat[10],"\n")
# print(y_train[10])

model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])


model.fit(x_train_flat,y_train,epochs=1)