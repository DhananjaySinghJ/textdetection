import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Example function to load and preprocess images (frames)
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize to match input size of CNN
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load data (Assuming you have folders 'texted', 'semi-textless', and 'textless')
texted_images, texted_labels = load_images_from_folder('texted', 2)
semi_textless_images, semi_textless_labels = load_images_from_folder('semi-textless', 1)
textless_images, textless_labels = load_images_from_folder('textless', 0)

# Combine datasets
X = np.concatenate((texted_images, semi_textless_images, textless_images), axis=0)
y = np.concatenate((texted_labels, semi_textless_labels, textless_labels), axis=0)

# Normalize and split the dataset
X = X / 255.0  # Normalize pixel values
y = to_categorical(y, num_classes=3)  # One-hot encode labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save('frame_classifier.h5')
