import numpy as np
import os

# Define selected classes
classes = ["snake", "sun", "penguin", "house", "bus"]
data = []
labels = []

# Load each class from .npy files
for i, cls in enumerate(classes):
    file_path = f"{cls}.npy"  # Ensure these files are in the same directory
    if os.path.exists(file_path):
        images = np.load(file_path)
        images = images[:5000]  # Use only 5000 images per class for balance
        data.append(images)
        labels.append(np.full(len(images), i))  # Assign class labels (0,1,2...)

# Convert lists to NumPy arrays
X = np.concatenate(data, axis=0)  # Combine all images
y = np.concatenate(labels, axis=0)  # Combine all labels

print(f"Loaded {X.shape[0]} images of shape {X.shape[1:]}")  # Check dataset size


# Normalize pixel values to [0,1]
X = X.astype("float32") / 255.0

# Reshape for CNN (add a single channel for grayscale images)
X = X.reshape(-1, 28, 28, 1)

# Check new shape
print(f"Final dataset shape: {X.shape}")  # Should be (total_samples, 28, 28, 1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")


import tensorflow as tf
from tensorflow import keras

# Define input layer
inputs = keras.Input(shape=(28, 28, 1))

# Define CNN model layers
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x)

x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)

x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation='relu')(x)
outputs = keras.layers.Dense(5, activation='softmax')(x)  # 5 classes

# Create model
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model for later use
model.save("doodle_classifier.keras")


# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
