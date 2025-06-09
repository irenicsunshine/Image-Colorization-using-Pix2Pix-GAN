from flask import Flask, request, send_file, render_template, jsonify
import numpy as np
import cv2
import os
import torch
from io import BytesIO
import logging
from PIL import Image

# Import our Pix2Pix GAN model
from pix2pix_model import get_pretrained_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Use relative paths instead of hardcoded ones
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize the colorization model
colorization_model = None

def load_model():
    global colorization_model
    try:
        logger.info("Loading Pix2Pix GAN model...")
        colorization_model = get_pretrained_model()
        logger.info("Pix2Pix GAN model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.route('/')
def index():
    return render_template('color1.html')

@app.route('/check-model')
def check_model():
    # For the Pix2Pix GAN model
    model_path = os.path.join(BASE_DIR, "model/pix2pix_generator.pth")
    return jsonify({
        'status': 'ready',
        'pix2pix_model': os.path.exists(model_path),
        'note': 'Using Pix2Pix GAN model, pre-trained weights not required but recommended'
    })

@app.route('/colorize', methods=['POST'])
def colorize_image():
    # Check if model is loaded, try to load if not
    global colorization_model
    if colorization_model is None:
        if not load_model():
            return jsonify({
                'error': 'Model failed to load. Check server logs.'
            }), 500
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        
        try:
            # Open image with PIL
            img = Image.open(file)
            
            # Ensure image is in RGB mode
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
        except Exception as e:
            logger.error(f"Error opening image: {str(e)}")
            return jsonify({'error': 'Failed to open image'}), 400

        # Process the image using our PyTorch model
        logger.info("Processing image for colorization")
        
        try:
            # Colorize the image
            logger.info("Colorizing the image using Pix2Pix GAN model")
            colorized = colorization_model.colorize(img)
            
            # Convert numpy array to PIL Image
            colorized_img = Image.fromarray(colorized)
            
            # Save to BytesIO
            io_buf = BytesIO()
            colorized_img.save(io_buf, format='PNG')
            io_buf.seek(0)
            
            logger.info("Colorization completed successfully")
            return send_file(io_buf, mimetype='image/png')
            
        except Exception as e:
            logger.error(f"Error during colorization: {str(e)}")
            return jsonify({'error': 'Failed during image colorization process'}), 500
        
    except Exception as e:
        logger.error(f"Error during request processing: {str(e)}")
        return jsonify({'error': 'Internal server error during processing'}), 500

if __name__ == '__main__':
    # Create required directories
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    
    # Load model at startup
    load_model()
    
    # Start the Flask app on port 8080 to avoid conflicts with AirPlay on macOS
    app.run(host='0.0.0.0', port=8080)
