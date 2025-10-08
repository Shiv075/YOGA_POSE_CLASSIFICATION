import os
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# ✅ Lazy load the model after the first request to avoid Render timeout
model = None
class_names = ['.ipynb_checkpoints', 'Bridge-Pose', 'Child-Pose', 'Cobra-Pose',
               'Downward-Dog-Pose', 'Pigeon-Pose', 'Standing-Mountain-Pose',
               'Tree-Pose', 'Triangle-Pose', 'Warrior-Pose']


def load_model():
    global model
    if model is None:
        print("Loading model...")
        model = tf.keras.models.load_model("model.h5")
        print("Model loaded successfully.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        load_model()  # Load model only once when first needed
        if 'file' not in request.files:
            return "No file uploaded", 400

        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400

        # ✅ Preprocess image
        img = Image.open(file).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        # ✅ Predict
        prediction = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]

        return f"Predicted Yoga Pose: {predicted_class}"

    except Exception as e:
        return f"❌ Internal Error: {str(e)}", 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)









