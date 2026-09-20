import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Dataset information
# -------------------
# print("Training images:", x_train.shape)
# print("Training labels:", y_train.shape)

# print("Testing images:", x_test.shape)
# print("Testing labels:", y_test.shape)

# Display the first training image and its label
# ----------------------------------------------
# plt.imshow(x_train[0], cmap='gray')
# plt.title(f"Label: {y_train[0]}")
# plt.show()

# Normalize the data
# ------------------
x_train = x_train / 255.0
x_test = x_test / 255.0