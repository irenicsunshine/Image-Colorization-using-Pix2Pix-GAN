#!/usr/bin/env python3
"""
Synthetic Dataset Generator for Image Colorization Training

Since downloading Places365 failed, this script creates a synthetic dataset
of images that can be used for initial testing of the Pix2Pix GAN colorization model.

The dataset consists of:
1. Random patterns and gradients
2. Simple shapes and objects
3. Procedurally generated scenes

Usage:
    python generate_synthetic_dataset.py --output_dir dataset/synthetic --num_images 1000
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw
import random
from tqdm import tqdm
from pathlib import Path
import math


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for colorization")
    parser.add_argument("--output_dir", type=str, default="dataset/synthetic",
                      help="Output directory for dataset")
    parser.add_argument("--num_images", type=int, default=1000,
                      help="Number of images to generate")
    parser.add_argument("--val_split", type=float, default=0.1,
                      help="Percentage of validation images")
    parser.add_argument("--image_size", type=int, default=256,
                      help="Size of generated images (square)")
    return parser.parse_args()


def generate_random_color():
    """Generate a random RGB color"""
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )


def generate_gradient(size, horizontal=True):
    """Generate a random gradient image"""
    img = Image.new('RGB', (size, size), color=generate_random_color())
    draw = ImageDraw.Draw(img)
    
    start_color = generate_random_color()
    end_color = generate_random_color()
    
    for i in range(size):
        t = i / size
        r = int(start_color[0] * (1 - t) + end_color[0] * t)
        g = int(start_color[1] * (1 - t) + end_color[1] * t)
        b = int(start_color[2] * (1 - t) + end_color[2] * t)
        
        if horizontal:
            draw.line([(i, 0), (i, size)], fill=(r, g, b))
        else:
            draw.line([(0, i), (size, i)], fill=(r, g, b))
    
    return img


def generate_radial_gradient(size):
    """Generate a radial gradient"""
    center = (size // 2, size // 2)
    max_radius = size // 2
    
    start_color = generate_random_color()
    end_color = generate_random_color()
    
    img = Image.new('RGB', (size, size), color=(0, 0, 0))
    data = np.zeros((size, size, 3), dtype=np.uint8)
    
    for y in range(size):
        for x in range(size):
            # Calculate distance from center
            distance = math.sqrt((x - center[0])**2 + (y - center[1])**2)
            # Normalize
            t = min(1.0, distance / max_radius)
            
            # Interpolate color
            r = int(start_color[0] * (1 - t) + end_color[0] * t)
            g = int(start_color[1] * (1 - t) + end_color[1] * t)
            b = int(start_color[2] * (1 - t) + end_color[2] * t)
            
            data[y, x] = [r, g, b]
    
    img = Image.fromarray(data)
    return img


def generate_shapes(size):
    """Generate an image with random shapes"""
    bg_color = generate_random_color()
    img = Image.new('RGB', (size, size), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    num_shapes = random.randint(3, 15)
    
    for _ in range(num_shapes):
        shape_type = random.choice(['circle', 'rectangle', 'line', 'triangle'])
        color = generate_random_color()
        
        # Random position and size
        x1 = random.randint(0, size - 1)
        y1 = random.randint(0, size - 1)
        x2 = random.randint(x1, size - 1)
        y2 = random.randint(y1, size - 1)
        
        if shape_type == 'circle':
            radius = random.randint(5, size // 4)
            draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill=color)
        
        elif shape_type == 'rectangle':
            draw.rectangle([x1, y1, x2, y2], fill=color)
        
        elif shape_type == 'line':
            width = random.randint(1, 10)
            draw.line([x1, y1, x2, y2], fill=color, width=width)
        
        elif shape_type == 'triangle':
            x3 = random.randint(0, size - 1)
            y3 = random.randint(0, size - 1)
            draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=color)
    
    return img


def generate_pattern(size):
    """Generate a repeating pattern"""
    pattern_size = random.choice([8, 16, 32, 64])
    img = Image.new('RGB', (size, size), color=(0, 0, 0))
    
    for y in range(0, size, pattern_size):
        for x in range(0, size, pattern_size):
            color = generate_random_color()
            pattern_type = random.choice(['solid', 'checkerboard', 'stripes'])
            
            if pattern_type == 'solid':
                draw = ImageDraw.Draw(img)
                draw.rectangle([x, y, x + pattern_size, y + pattern_size], fill=color)
            
            elif pattern_type == 'checkerboard':
                draw = ImageDraw.Draw(img)
                alt_color = generate_random_color()
                half_size = pattern_size // 2
                
                draw.rectangle([x, y, x + half_size, y + half_size], fill=color)
                draw.rectangle([x + half_size, y + half_size, x + pattern_size, y + pattern_size], fill=color)
                draw.rectangle([x + half_size, y, x + pattern_size, y + half_size], fill=alt_color)
                draw.rectangle([x, y + half_size, x + half_size, y + pattern_size], fill=alt_color)
            
            elif pattern_type == 'stripes':
                draw = ImageDraw.Draw(img)
                alt_color = generate_random_color()
                stripe_width = pattern_size // 4
                
                for i in range(0, pattern_size, stripe_width * 2):
                    draw.rectangle([x + i, y, x + i + stripe_width, y + pattern_size], fill=color)
                    draw.rectangle([x + i + stripe_width, y, x + i + stripe_width * 2, y + pattern_size], fill=alt_color)
    
    return img


def generate_scene(size):
    """Generate a simple scene with sky, ground, and objects"""
    img = Image.new('RGB', (size, size), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Sky
    sky_color = (random.randint(100, 200), random.randint(150, 255), random.randint(200, 255))
    draw.rectangle([0, 0, size, size * 2 // 3], fill=sky_color)
    
    # Ground
    ground_color = (random.randint(50, 150), random.randint(100, 200), random.randint(0, 100))
    draw.rectangle([0, size * 2 // 3, size, size], fill=ground_color)
    
    # Sun or moon
    if random.random() < 0.7:  # 70% chance of having sun/moon
        sun_radius = random.randint(size // 10, size // 5)
        sun_x = random.randint(sun_radius, size - sun_radius)
        sun_y = random.randint(sun_radius, size * 2 // 3 - sun_radius)
        sun_color = (random.randint(200, 255), random.randint(200, 255), random.randint(0, 150))
        draw.ellipse((sun_x - sun_radius, sun_y - sun_radius, 
                      sun_x + sun_radius, sun_y + sun_radius), fill=sun_color)
    
    # Buildings or trees
    num_objects = random.randint(2, 6)
    for i in range(num_objects):
        obj_type = random.choice(['building', 'tree'])
        obj_x = i * size // num_objects + random.randint(-20, 20)
        
        if obj_type == 'building':
            obj_width = random.randint(size // 12, size // 6)
            obj_height = random.randint(size // 6, size // 3)
            obj_y = size * 2 // 3 - obj_height
            obj_color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
            draw.rectangle([obj_x, obj_y, obj_x + obj_width, size * 2 // 3], fill=obj_color)
            
            # Windows
            window_count = random.randint(2, 6)
            for w in range(window_count):
                window_x = obj_x + obj_width * w // window_count + 2
                window_y = obj_y + random.randint(5, obj_height // 2)
                window_size = max(min(obj_width // window_count - 4, 10), 2)  # Ensure size is at least 2
                # Ensure window is inside building
                window_x = min(window_x, obj_x + obj_width - window_size - 1)
                draw.rectangle([window_x, window_y, window_x + window_size, window_y + window_size], 
                              fill=(255, 255, 200))
        
        elif obj_type == 'tree':
            trunk_width = random.randint(size // 40, size // 20)
            trunk_height = random.randint(size // 10, size // 5)
            trunk_y = size * 2 // 3 - trunk_height
            trunk_color = (random.randint(50, 100), random.randint(30, 60), random.randint(0, 30))
            draw.rectangle([obj_x, trunk_y, obj_x + trunk_width, size * 2 // 3], fill=trunk_color)
            
            # Foliage
            foliage_radius = random.randint(size // 15, size // 8)
            foliage_y = trunk_y
            foliage_color = (random.randint(0, 100), random.randint(100, 200), random.randint(0, 100))
            draw.ellipse((obj_x - foliage_radius + trunk_width // 2, 
                         foliage_y - foliage_radius,
                         obj_x + foliage_radius + trunk_width // 2, 
                         foliage_y + foliage_radius), fill=foliage_color)
    
    return img


def generate_random_image(size):
    """Generate a random synthetic image"""
    generators = [
        generate_gradient,
        lambda s: generate_gradient(s, horizontal=False),
        generate_radial_gradient,
        generate_shapes,
        generate_pattern,
        generate_scene
    ]
    
    generator = random.choice(generators)
    return generator(size)


def main():
    args = parse_args()
    
    # Create output directories
    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Calculate split
    val_count = int(args.num_images * args.val_split)
    train_count = args.num_images - val_count
    
    # Generate training images
    print(f"Generating {train_count} training images...")
    for i in tqdm(range(train_count)):
        img = generate_random_image(args.image_size)
        img.save(os.path.join(train_dir, f"{i:05d}.jpg"))
    
    # Generate validation images
    print(f"Generating {val_count} validation images...")
    for i in tqdm(range(val_count)):
        img = generate_random_image(args.image_size)
        img.save(os.path.join(val_dir, f"{i:05d}.jpg"))
    
    # Return info for training
    train_images = list(Path(train_dir).glob("*.jpg"))
    val_images = list(Path(val_dir).glob("*.jpg"))
    
    print(f"Dataset created with {len(train_images)} training images and {len(val_images)} validation images")
    print(f"Dataset location: {os.path.abspath(args.output_dir)}")
    
    return {
        "train_dir": train_dir,
        "val_dir": val_dir,
        "train_images": len(train_images),
        "val_images": len(val_images)
    }


if __name__ == "__main__":
    main()
