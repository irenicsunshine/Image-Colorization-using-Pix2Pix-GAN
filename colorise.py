 #!/usr/bin/env python3
"""
Image Colorization CLI Tool
Using Pix2Pix GAN colorization model

Usage:
    python colorise.py -i path/to/image.jpg [-o path/to/output.jpg] [--display]
"""

import argparse
import os
import sys
import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import our Pix2Pix GAN model
from pix2pix_model import get_pretrained_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Use relative paths instead of hardcoded ones
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(description='Image Colorization Tool')
    ap.add_argument("-i", "--image", type=str, required=True,
                    help="path to input black and white image")
    ap.add_argument("-o", "--output", type=str,
                    help="path to output colorized image (optional)")
    ap.add_argument("--display", action="store_true",
                    help="display result image (requires GUI)")
    return vars(ap.parse_args())


def load_model():
    """Load the Pix2Pix GAN colorization model."""
    try:
        logger.info("Loading Pix2Pix GAN colorization model...")
        model = get_pretrained_model()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def colorize_image(model, image_path):
    """Colorize a grayscale image using Pix2Pix GAN model."""
    try:
        # Read the image using PIL
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Store the original for comparison
        original = np.array(img)
        
        logger.info("Colorizing the image...")
        colorized = model.colorize(img)
        
        logger.info("Colorization completed")
        return original, colorized
    except Exception as e:
        logger.error(f"Error during colorization: {str(e)}")
        return None, None


def main():
    # Parse command line arguments
    args = parse_args()
    image_path = args["image"]
    output_path = args.get("output")
    display_result = args.get("display", False)
    
    # Check if image exists
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        sys.exit(1)
    
    # Load the model
    model = load_model()
    if model is None:
        sys.exit(1)
    
    # Colorize the image
    original, colorized = colorize_image(model, image_path)
    if colorized is None:
        sys.exit(1)
    
    # Save the result if output path is specified
    if output_path:
        logger.info(f"Saving colorized image to {output_path}")
        Image.fromarray(colorized).save(output_path)
        logger.info(f"Saved to {output_path}")
    
    # Display the result if requested
    if display_result:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(colorized)
        plt.title('Colorized')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()