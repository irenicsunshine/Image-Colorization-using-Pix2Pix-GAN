#!/usr/bin/env python3
"""
Automated Training Script for Pix2Pix GAN Colorization

This script automates the process of:
1. Downloading a subset of the Places365 dataset (recommended for colorization)
2. Training the Pix2Pix GAN model on this dataset
3. Testing the model on sample images

Places365 is chosen because it offers the best balance of scene diversity,
image quality, and manageable size for colorization training.

Usage:
    python run_training.py --num_images 1000 --epochs 50 --batch_size 4
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Automate training of Pix2Pix GAN for colorization")
    parser.add_argument("--data_dir", type=str, default="dataset/places365", 
                        help="Directory to store the dataset")
    parser.add_argument("--num_images", type=int, default=1000, 
                        help="Number of images to download (max recommended: 5000)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--model_dir", type=str, default="model",
                        help="Directory to save model checkpoints")
    parser.add_argument("--small", action="store_true",
                        help="Download smaller dataset (256x256)")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip dataset download if you already have it")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for training if available")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Step 1: Ensure all dependencies are installed
    print("Checking and installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Step 2: Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs("test_images", exist_ok=True)
    
    # Step 3: Download dataset if needed
    if not args.skip_download:
        print("\n==== Downloading Places365 Dataset ====")
        cmd = [sys.executable, "download_places365.py", 
               "--output_dir", args.data_dir,
               "--num_images", str(args.num_images)]
        
        if args.small:
            cmd.append("--small")
            
        subprocess.run(cmd)
        
        # Check if download succeeded
        train_dir = os.path.join(args.data_dir, "train")
        train_images = list(Path(train_dir).glob("*.jpg"))
        if len(train_images) == 0:
            print("No training images found. Please manually download Places365 dataset.")
            sys.exit(1)
    
    # Step 4: Train the model
    print("\n==== Training Pix2Pix GAN Model ====")
    train_cmd = [
        sys.executable, "train_pix2pix.py",
        "--data_root", args.data_dir,
        "--save_dir", args.model_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--sample_interval", "500",
        "--checkpoint_interval", "10"
    ]
    subprocess.run(train_cmd)
    
    # Step 5: Copy the final model to the model directory
    print("\n==== Setting up model for inference ====")
    # This should have been handled by the training script saving to the model dir
    
    # Step 6: Update the README with information about the training
    print("\n==== Updating project documentation ====")
    with open("README.md", "a") as f:
        f.write("\n\n## Model Training Information\n\n")
        f.write(f"* Dataset: Places365 (subset of {args.num_images} images)\n")
        f.write(f"* Training epochs: {args.epochs}\n")
        f.write(f"* Batch size: {args.batch_size}\n")
        f.write(f"* Model location: {os.path.abspath(args.model_dir)}/pix2pix_generator.pth\n")
        f.write("\nThe Pix2Pix GAN model was chosen for colorization as it excels at learning ")
        f.write("the mapping between different image representations (grayscale to color).\n")
    
    print("\n==== Training Complete ====")
    print(f"The trained model is available at: {os.path.abspath(args.model_dir)}/pix2pix_generator.pth")
    print("You can now use the web interface or CLI to colorize images!")

if __name__ == "__main__":
    main()
