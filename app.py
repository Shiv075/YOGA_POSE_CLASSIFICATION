from flask import Flask, request, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import base64

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('yoga_model.h5')

# Class names
class_names = ['.ipynb_checkpoints', 'Bridge-Pose', 'Child-Pose', 'Cobra-Pose',
               'Downward-Dog-Pose', 'Pigeon-Pose', 'Standing-Mountain-Pose',
               'Tree-Pose', 'Triangle-Pose', 'Warrior-Pose']

# Function to load and preprocess image
def load_and_resize_image(img, target_size=(224, 224)):
    img = Image.open(img).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # normalize
    return img_array

# Convert image to base64 for display
def img_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded'
    file = request.files['file']
    if file.filename == '':
        return 'No file selected'

    img = load_and_resize_image(file)
    pred = model.predict(tf.expand_dims(img, axis=0))
    predicted_class = class_names[pred[0].argmax()]

    # Reset file pointer to read image for display
    file.seek(0)
    img_display = Image.open(file)
    img_base64 = img_to_base64(img_display)

    return render_template('result.html', prediction=predicted_class, img_data=img_base64)

import os

if __name__ == "__main__":
    # Get port from environment variable or default to 5000
    port = int(os.environ.get("PORT", 5000))
    
    # Print info to logs (helps debugging on Render)
    print(f"Starting server on port {port}...")
    
    # IMPORTANT: Bind to 0.0.0.0 so Render detects the port
    app.run(host="0.0.0.0", port=port, debug=False)






