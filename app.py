from flask import Flask, Response, render_template, request, jsonify
from utils import processing
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

uploaded_image = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global uploaded_image
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."})

    uploaded_file = request.files['image']

    if uploaded_file.filename == '':
        return jsonify({"error": "No selected file."})

    uploaded_image = uploaded_file.read()

    return jsonify({"image": "/uploaded_image"})

@app.route('/uploaded_image')
def get_uploaded_image():
    global uploaded_image
    if uploaded_image is None:
        return jsonify({"error": "No image uploaded yet."})

    return Response(uploaded_image, mimetype='image/jpeg')

@app.route('/classify', methods=['POST'])
def classify():
    global uploaded_image
    if uploaded_image is None:
        return jsonify({"error": "Please upload an image first."})

    result = processing.image_classifier(uploaded_image)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)