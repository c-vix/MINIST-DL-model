import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("mnist_model.keras")

# Load test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_test = x_test / 255.0

# Select one image
image = x_test[0]

# Add batch dimension
image_batch = np.expand_dims(image, axis=0)

# Predict
prediction = model.predict(image_batch)

predicted_digit = np.argmax(prediction)

print("Predicted digit:", predicted_digit)
print("Actual digit:", y_test[0])

# Display image
plt.imshow(image, cmap="gray")
plt.title(f"Predicted: {predicted_digit}")
plt.show()