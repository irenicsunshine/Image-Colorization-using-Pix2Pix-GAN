#!/usr/bin/env python3
"""
Places365 Dataset Downloader for Image Colorization

This script downloads a subset of the Places365-Standard dataset for training
the Pix2Pix GAN colorization model.

Places365 is ideal for colorization because it contains diverse scenes and
environments, giving the model exposure to a wide range of contexts.

Usage:
    python download_places365.py --output_dir dataset --num_images 5000
"""

import os
import argparse
import requests
from tqdm import tqdm
import random
import zipfile
from PIL import Image
import io
import multiprocessing
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Download Places365 dataset for colorization")
    parser.add_argument("--output_dir", type=str, default="dataset/places365", help="Output directory for downloaded data")
    parser.add_argument("--num_images", type=int, default=5000, 
                        help="Number of images to download (max 5000 for standard version)")
    parser.add_argument("--val_split", type=float, default=0.1, 
                        help="Percentage of images to use for validation")
    parser.add_argument("--small", action="store_true", 
                        help="Download the small version (256x256) instead of standard")
    return parser.parse_args()


def download_file(url, output_path):
    """Download a single file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {os.path.basename(output_path)}") as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    return output_path


def download_places365_categories():
    """Download the Places365 categories list"""
    categories_url = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"
    response = requests.get(categories_url)
    categories = []
    
    if response.status_code == 200:
        for line in response.text.strip().split('\n'):
            if line:
                category = line.strip().split(' ')[0]
                # Remove /c/ prefix if present
                if category.startswith('/'):
                    category = category[1:]
                categories.append(category)
        print(f"Downloaded {len(categories)} categories")
        return categories
    else:
        print(f"Failed to download categories: {response.status_code}")
        # Return a few common categories as fallback
        return ["kitchen", "bedroom", "mountain", "beach", "office", "street"]


def download_places365_dev_set():
    """Download Places365 development kit with URLs"""
    dev_url = "http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar"
    output_path = "places365_filelist.tar"
    
    try:
        download_file(dev_url, output_path)
        
        # Extract the tar file
        print("Extracting file list...")
        import tarfile
        with tarfile.open(output_path) as tar:
            tar.extractall()
        
        # Read the training URLs
        train_urls_file = "places365_train_standard.txt"
        if os.path.exists(train_urls_file):
            with open(train_urls_file, 'r') as f:
                urls = [line.strip() for line in f.readlines()]
            print(f"Found {len(urls)} training URLs")
            return urls
        else:
            print(f"Could not find {train_urls_file}")
            return []
            
    except Exception as e:
        print(f"Error downloading development set: {e}")
        return []
    finally:
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)


def download_and_verify_image(args):
    """Download and verify a single image"""
    url, output_path = args
    try:
        # Add base URL if needed
        if not url.startswith('http'):
            url = f"http://places2.csail.mit.edu/imgs/256/" + url
            
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            try:
                # Verify it's a valid image
                img = Image.open(io.BytesIO(response.content))
                img.verify()  # Verify it's a valid image
                
                # Save the image
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return True
            except Exception:
                return False
        return False
    except Exception:
        return False


def download_images_from_urls(urls, output_dir, num_images, val_split=0.1):
    """Download images from URLs and split into train/val sets"""
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Shuffle and limit the number of URLs
    random.shuffle(urls)
    urls = urls[:num_images]
    
    # Calculate split
    val_count = int(num_images * val_split)
    train_count = num_images - val_count
    
    train_urls = urls[:train_count]
    val_urls = urls[train_count:num_images]
    
    # Prepare arguments for multiprocessing
    train_args = [(url, os.path.join(train_dir, f"{i:05d}.jpg")) 
                 for i, url in enumerate(train_urls)]
    val_args = [(url, os.path.join(val_dir, f"{i:05d}.jpg")) 
               for i, url in enumerate(val_urls)]
    
    # Download train images
    print(f"Downloading {len(train_args)} training images...")
    with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), 8)) as pool:
        results = list(tqdm(pool.imap(download_and_verify_image, train_args), total=len(train_args)))
    train_success = sum(results)
    
    # Download validation images
    print(f"Downloading {len(val_args)} validation images...")
    with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), 8)) as pool:
        results = list(tqdm(pool.imap(download_and_verify_image, val_args), total=len(val_args)))
    val_success = sum(results)
    
    print(f"Successfully downloaded {train_success}/{len(train_args)} training images")
    print(f"Successfully downloaded {val_success}/{len(val_args)} validation images")


def download_places365_small(output_dir, num_images=5000, val_split=0.1):
    """Download the small version (256x256) of Places365"""
    # URLs for the small version archives
    val_url = "http://data.csail.mit.edu/places/places365/val_256.tar"
    train_url = "http://data.csail.mit.edu/places/places365/train_256_places365standard.tar"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # We'll use a different approach for small version
    # Since downloading individual images can be slow, we'll download validation set
    # which is smaller and extract a subset
    print("Downloading validation set (this might take a while)...")
    val_path = os.path.join(output_dir, "val_256.tar")
    
    try:
        download_file(val_url, val_path)
        
        # Extract a subset of images
        print("Extracting images...")
        train_dir = os.path.join(output_dir, "train")
        val_dir = os.path.join(output_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Calculate split
        val_count = int(num_images * val_split)
        train_count = num_images - val_count
        total_count = 0
        
        with tarfile.open(val_path) as tar:
            members = tar.getmembers()
            random.shuffle(members)
            
            for i, member in enumerate(tqdm(members)):
                if not member.isfile() or not member.name.endswith(('.jpg', '.png', '.jpeg')):
                    continue
                
                if total_count < train_count:
                    target_dir = train_dir
                elif total_count < num_images:
                    target_dir = val_dir
                else:
                    break
                
                # Extract the file with a new name
                f = tar.extractfile(member)
                if f is not None:
                    content = f.read()
                    new_name = f"{total_count:05d}.jpg"
                    with open(os.path.join(target_dir, new_name), 'wb') as out_file:
                        out_file.write(content)
                    total_count += 1
                    
        print(f"Successfully extracted {total_count} images")
        
    except Exception as e:
        print(f"Error in download: {e}")
    finally:
        # Clean up
        if os.path.exists(val_path):
            os.remove(val_path)


def generate_random_images(output_dir, num_images=5000, val_split=0.1):
    """Generate random synthetic images if download fails"""
    print("Generating synthetic images as fallback...")
    
    # Create directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Calculate splits
    val_count = int(num_images * val_split)
    train_count = num_images - val_count
    
    # Generate training images
    for i in tqdm(range(train_count), desc="Generating training images"):
        img = Image.new('RGB', (256, 256), color=(
            random.randint(0, 255), 
            random.randint(0, 255), 
            random.randint(0, 255)
        ))
        img.save(os.path.join(train_dir, f"{i:05d}.jpg"))
    
    # Generate validation images
    for i in tqdm(range(val_count), desc="Generating validation images"):
        img = Image.new('RGB', (256, 256), color=(
            random.randint(0, 255), 
            random.randint(0, 255), 
            random.randint(0, 255)
        ))
        img.save(os.path.join(val_dir, f"{i:05d}.jpg"))
    
    print(f"Generated {train_count} training images and {val_count} validation images")


def main():
    """Main function to download Places365 dataset"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Try downloading URLs first
    print("Trying to download Places365 file list...")
    urls = download_places365_dev_set()
    
    # If we got URLs, download individual images
    if urls:
        download_images_from_urls(urls, args.output_dir, args.num_images, args.val_split)
    else:
        # Try downloading the small version
        if args.small:
            print("Downloading Places365 small version...")
            download_places365_small(args.output_dir, args.num_images, args.val_split)
        else:
            # Fallback to generating random images
            print("Could not download Places365 dataset. Generating synthetic images instead.")
            print("For real images, please manually download Places365 from:")
            print("http://places2.csail.mit.edu/download.html")
            generate_random_images(args.output_dir, args.num_images, args.val_split)
    
    # Check if we have any images
    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "val")
    train_images = list(Path(train_dir).glob("*.jpg"))
    val_images = list(Path(val_dir).glob("*.jpg"))
    
    print(f"Dataset created with {len(train_images)} training images and {len(val_images)} validation images")
    print(f"Dataset location: {os.path.abspath(args.output_dir)}")
    
    if len(train_images) == 0 and len(val_images) == 0:
        print("No images were downloaded. Please manually download Places365 from:")
        print("http://places2.csail.mit.edu/download.html")
        
    # Return info for training
    return {
        "train_dir": train_dir,
        "val_dir": val_dir,
        "train_images": len(train_images),
        "val_images": len(val_images)
    }


if __name__ == "__main__":
    main()
