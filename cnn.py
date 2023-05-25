import tensorflow as tf
from tensorflow.keras import datasets,layers,models
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# plt.figure(figsize=(10,4))
print(plt.imshow(x_test[6]))

x_train = x_train/255
x_test = x_test/255

x_train_flat = x_train.reshape(len(x_train), 32*32*3)
x_test_flat = x_test.reshape(len(x_test), 32*32*3)

# print(len(x_train))
# print(len(x_test))


model= models.Sequential([
      layers.Conv2D(filters=30, kernel_size=(3,3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(filters=30, kernel_size=(3,3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(filters=30, kernel_size=(3,3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Flatten(),
      layers.Dense(3000, input_shape=(3072,), activation='relu'),
      layers.Dense(1000, input_shape=(3072,), activation='relu'),
      layers.Dense(500, input_shape=(3072,), activation='relu'),
      layers.Dense(10, input_shape=(3072,), activation='sigmoid')
])
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=1)