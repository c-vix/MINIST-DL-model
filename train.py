import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
# --------------------

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

# Building the model
# ------------------

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') 
])

# Compile the model
# -----------------

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
# -----------------

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# 6. Evaluate model
# --------------------------------

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# 7. Save model
# --------------------------------

model.save("mnist_model.keras")

print("Model saved!")