# Image Colorization Application

This application uses deep learning to automatically colorize black and white or grayscale images. It's built with a Flask backend and a simple, intuitive web interface.

## Features

- Upload black and white or grayscale images through a web interface
- Automatic colorization using a pre-trained neural network
- Download colorized results
- Available as both web application and command-line tool

## Getting Started

### Prerequisites

- Python 3.6+
- OpenCV
- Flask
- NumPy

### Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the pre-trained model files (these are required to run the application)

```bash
# Create model directory if it doesn't exist
mkdir -p model

# Download model files
wget -O model/colorization_deploy_v2.prototxt https://raw.githubusercontent.com/richzhang/colorization/master/colorization/models/colorization_deploy_v2.prototxt
wget -O model/pts_in_hull.npy https://raw.githubusercontent.com/richzhang/colorization/master/colorization/resources/pts_in_hull.npy
wget -O model/colorization_release_v2.caffemodel http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel
```

### Usage

#### Web Application

1. Start the Flask server:

```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`
3. Upload a grayscale image, click "Colorize Image", and download or share the result

#### Command Line Tool

To colorize an image from the command line:

```bash
python colorise.py --image path/to/your/image.jpg --output path/to/save/result.jpg
```

Add the `--display` flag to show the original and colorized images:

```bash
python colorise.py --image path/to/your/image.jpg --display
```

## How It Works

This application uses a pre-trained deep neural network that can predict the chrominance (color) values for each pixel in a grayscale image. The model was trained on a large dataset of color images and learned to associate specific textures, shapes, and intensities with appropriate colors.

Technically, the process works by:

1. Converting the input image to LAB color space
2. Using the L channel (lightness) as input to the neural network
3. Predicting the a and b channels (which contain the color information)
4. Combining the original L channel with the predicted a and b channels
5. Converting the result back to RGB color space

## Citation

This implementation is based on the work from:

```
Zhang, R., Isola, P., & Efros, A. A. (2016). Colorful image colorization. 
In European conference on computer vision (pp. 649-666). Springer, Cham.
```

The original implementation and more details can be found at: https://github.com/richzhang/colorization

## License

This project is licensed under the MIT License - see the LICENSE file for details.
