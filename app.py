from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from tensorflow import keras
from PIL import Image
import io
import os

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

MODEL_PATH = r"models/fruit_freshness_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy model tại: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)
print(" Model loaded successfully!")

LABELS = [
    'freshapples', 'freshbanana', 'freshcucumber', 'freshokra', 'freshoranges', 'freshpotato', 'freshtomato',
    'rottenapples', 'rottenbanana', 'rottencucumber', 'rottenokra', 'rottenoranges', 'rottenpotato', 'rottentomato'
]


def preprocess_image(image_data):
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Không tìm thấy ảnh trong request'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Chưa chọn file ảnh'}), 400

        image_data = file.read()
        img_array = preprocess_image(image_data)
        print(" Ảnh sau khi xử lý:", img_array.shape)

        prediction = model.predict(img_array)
        print(" Raw prediction output:", repr(prediction))

        pred_arr = np.array(prediction)
        if pred_arr.size == 0:
            return jsonify({'error': 'Model trả về mảng rỗng'}), 500

        if pred_arr.ndim >= 2 and pred_arr.shape[1] >= 2:
            probs = pred_arr[0]
            idx = int(np.argmax(probs))
            if len(LABELS) != probs.shape[0]:
                return jsonify({
                    'error': f'Số lượng lớp trong LABELS ({len(LABELS)}) không khớp output model ({probs.shape[0]})',
                    'raw_prediction': probs.tolist()
                }), 500

            label = LABELS[idx]
            confidence = round(float(probs[idx]) * 100, 2)
            probabilities = {LABELS[i]: round(float(probs[i]) * 100, 2) for i in range(probs.shape[0])}

        else:
            return jsonify({'error': 'Output model không đúng định dạng'}), 500

        result = {'label': label, 'confidence': confidence, 'probabilities': probabilities}
        print(" Predict result:", result)
        return jsonify(result)

    except Exception as e:
        import traceback
        print(" Exception in /predict:\n", traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print(" Flask server running at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
