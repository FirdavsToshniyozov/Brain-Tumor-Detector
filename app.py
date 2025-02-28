import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template

# Flask ilovasi
app = Flask(__name__)

# Modelni yuklash
model = tf.keras.models.load_model("brain_tumor_model.h5")

# Kategoriya nomlari
categories = ["Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor", "No Tumor"]

# Rasmni tahlil qilish funksiyasi
def predict_tumor(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))
    img = img.reshape(1, 150, 150, 1) / 255.0  # Normalizatsiya
    prediction = model.predict(img)
    class_index = np.argmax(prediction)  # Eng katta ehtimollik
    return categories[class_index], prediction[0][class_index]

# Asosiy sahifa
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return "Fayl yuklanmadi!"

        file = request.files["file"]
        if file.filename == "":
            return "Fayl tanlanmadi!"

        file_path = os.path.join("static/uploads", file.filename)
        file.save(file_path)

        # Model orqali tahlil qilish
        result, confidence = predict_tumor(file_path)

        return render_template("index.html", image=file_path, result=result, confidence=confidence)

    return render_template("index.html", image=None, result=None, confidence=None)

# Flask serverni ishga tushirish
if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)  # Fayllarni saqlash uchun papka yaratish
    app.run(debug=True)
