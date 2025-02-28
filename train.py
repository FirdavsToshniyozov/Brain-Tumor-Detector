import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

# Dataset joylashuvi
dataset_path = "dataset/Training"
categories = ["glioma", "meningioma", "pituitary", "notumor"]

# Datasetni yuklash
X, y = [], []
for category in categories:
    path = os.path.join(dataset_path, category)
    label = categories.index(category)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (150, 150))
        X.append(img)
        y.append(label)

# Convert to numpy array
X = np.array(X).reshape(-1, 150, 150, 1) / 255.0
y = np.array(y)

# CNN model yaratish
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')  # 4 ta sinf
])

# Modelni kompilyatsiya qilish
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modelni o‘qitish
model.fit(X, y, epochs=10, validation_split=0.2)

# Modelni saqlash
model.save("brain_tumor_model.h5")
print("✅ Model saqlandi!")
