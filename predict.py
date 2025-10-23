import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

MODEL_PATH = "models/fruit_freshness_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)

class_names = sorted(os.listdir("dataset/Train"))

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    print(f"Ảnh: {os.path.basename(img_path)}")
    print(f"➡ Loại quả dự đoán: {predicted_class}")
    print(f"➡ Độ tin cậy: {confidence:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    test_image = "sample_test/banana.jpg"
    predict_image(test_image)
