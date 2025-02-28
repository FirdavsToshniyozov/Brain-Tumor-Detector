import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# CNN Model yaratish
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),  # Tasvir kiritish qismi
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary Classification (Tumor bor/yok)
])

# Modelni kompilyatsiya qilish
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modelni saqlash
model.save('brain_tumor_model.h5')

print("Model muvaffaqiyatli saqlandi!")
