from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = load_model("model.h5")

# Class labels (update if your class_map is different)
class_map = {
    0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3: 'Brinjal', 4: 'Broccoli',
    5: 'Cabbage', 6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower', 9: 'Cucumber',
    10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13: 'Radish', 14: 'Tomato'
}

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        file_path = os.path.join('static/uploads', file.filename)
        file.save(file_path)

        # Preprocess image
        img = image.load_img(file_path, target_size=(150, 150))
        img_arr = image.img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        prediction = model.predict(img_arr)
        predicted_label = np.argmax(prediction)
        result = class_map[predicted_label]

        return jsonify({'prediction': result, 'image_path': file_path})
    return jsonify({'error': 'No file uploaded'})

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
