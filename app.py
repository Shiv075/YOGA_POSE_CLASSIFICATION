import os
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

# ✅ Lazy load the model after the first request to avoid Render timeout
model = None
CLASS_NAMES = ['.ipynb_checkpoints', 'Bridge-Pose', 'Child-Pose', 'Cobra-Pose',
               'Downward-Dog-Pose', 'Pigeon-Pose', 'Standing-Mountain-Pose',
               'Tree-Pose', 'Triangle-Pose', 'Warrior-Pose']

# ✅ Load pose → image links mapping
pose_links = {}
pose_links_path = "pose_links.json"  # Make sure this JSON file is in the same folder
if os.path.exists(pose_links_path):
    with open(pose_links_path, 'r') as f:
        pose_links = json.load(f)
else:
    print("⚠️ pose_links.json not found. Images will not be displayed.")

def load_model():
    global model
    if model is None:
        print("Loading model...")
        model = tf.keras.models.load_model("yoga_model.h5")
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

        # ✅ Get corresponding image URL from JSON mapping
        img_url = pose_links.get(predicted_class, None)
        if isinstance(img_url, list):
            # If multiple images exist, pick the first one
            img_url = img_url[0] if len(img_url) > 0 else None

        return render_template('result.html', prediction=predicted_class, img_url=img_url)

    except Exception as e:
        return f"❌ Internal Error: {str(e)}", 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
