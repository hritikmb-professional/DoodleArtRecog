import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("doodle_classifier_improved.keras")

# Define image properties (ensure it matches training data)
IMAGE_WIDTH = 28  # Match your training input size
IMAGE_HEIGHT = 28

# Define class labels (update based on your dataset)
class_labels = ["snake", "bus", "penguin", "house", "sun"]  # Modify with actual class names

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale since your model expects (28, 28, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMAGE_WIDTH, IMAGE_HEIGHT))  # Resize to (28, 28)
    normalized = resized / 255.0  # Normalize
    input_image = np.expand_dims(normalized, axis=0)  # Add batch dimension
    input_image = np.expand_dims(input_image, axis=-1)  # Convert (28, 28) to (28, 28, 1)

    # Predict
    predictions = model.predict(input_image)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Display the result
    label = f"{class_labels[predicted_class]} ({confidence:.2f})"
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Doodle Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
