import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("doodle_classifier_improved.keras")

# Define image properties
IMAGE_SIZE = 28  # Match the training data size

# Class labels (must match training order)
class_labels = ["snake", "sun", "penguin", "house", "bus"]

# Load and preprocess the test image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))  # Resize to 28x28
    image = image.astype("float32") / 255.0  # Normalize
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Test an image
image_path = "test.jpg"  # Change this to your test image
input_image = preprocess_image(image_path)

# Predict using the trained model
predictions = model.predict(input_image)
predicted_class = np.argmax(predictions)
confidence = np.max(predictions)

# Ensure class_labels list does not cause index error
if predicted_class < len(class_labels):
    label = f"{class_labels[predicted_class]} ({confidence:.2f})"
else:
    label = f"Unknown Class ({confidence:.2f})"

print(f"Predicted Class: {label}")
