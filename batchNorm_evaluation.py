"""
Mission:
===========
“A Batch Normalization can improve a given model results.”
Please examine this argument by piece of Python code.
In this case, you can take any dataset and model from the internet 
(copy & paste, make it simple) and show the benefits of using it (via evaluation).
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Load data
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

# Base Model - No Batch Normalization
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

opt = optimizers.SGD(lr=0.1, momentum=0.9)
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x=train_images, y=train_labels, validation_split=0.2, epochs=100)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print("No BatchNormalization, learning rate 0.1, 100 epochs")
print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)


# With Batch Normaliozation
bn_model = Sequential()
bn_model.add(Flatten(input_shape=(28, 28)))
bn_model.add(Dense(128, activation='relu'))
bn_model.add(BatchNormalization())
bn_model.add(Dense(10, activation='softmax'))

opt = optimizers.SGD(lr=0.1, momentum=0.9)
bn_model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

bn_history = bn_model.fit(x=train_images, y=train_labels, validation_split=0.2, epochs=100)

test_loss, test_acc = bn_model.evaluate(test_images,  test_labels, verbose=2)

print("With BatchNormalization, learning rate 0.1, 100 epochs")
print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)

# plot history
print(history.history.keys())
plt.plot(bn_history.history['acc'], label='BatchNorm')
plt.plot(history.history['acc'], label='Standart')
plt.title("Accuracy, learning_rate=0.1, 100 epochs")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
