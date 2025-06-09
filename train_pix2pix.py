#!/usr/bin/env python3
"""
Pix2Pix GAN Training Script for Image Colorization

This script trains a Pix2Pix GAN model for colorizing grayscale images.
It uses a dataset of color images, converting them to LAB color space
and training the model to predict the ab channels from the L channel.

Usage:
    python train_pix2pix.py --data_root path/to/dataset --batch_size 4 --epochs 200
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
import time
from datetime import datetime

# Import our Pix2Pix GAN model
from pix2pix_model import Generator, Discriminator


class ColorizationDataset(Dataset):
    """Dataset for training the Pix2Pix colorization model"""
    def __init__(self, root_dir, transform=None, size=256):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on images.
            size (int): Size to resize images to
        """
        self.root_dir = root_dir
        self.transform = transform
        self.size = size
        
        # Get all image files
        valid_extensions = ('.jpg', '.jpeg', '.png')
        self.image_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(root_dir)
            for file in files if file.lower().endswith(valid_extensions)
        ]
        print(f"Found {len(self.image_paths)} images in {root_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        try:
            img_path = self.image_paths[idx]
            img = Image.open(img_path).convert('RGB')
            
            # Apply transforms if any
            if self.transform:
                img = self.transform(img)
            else:
                # Default transform
                img = transforms.Resize((self.size, self.size))(img)
                img = transforms.ToTensor()(img)
            
            # Convert to LAB
            img_lab = rgb2lab(img.permute(1, 2, 0).numpy())
            
            # Extract L and ab channels
            L = img_lab[:, :, 0]
            ab = img_lab[:, :, 1:]
            
            # Normalize L from [0, 100] to [-1, 1]
            L = (L - 50) / 50
            
            # Normalize ab from [-128, 127] to [-1, 1]
            ab = ab / 128
            
            # Convert back to tensors
            L_tensor = torch.from_numpy(L).unsqueeze(0).float()
            ab_tensor = torch.from_numpy(ab.transpose((2, 0, 1))).float()
            
            return L_tensor, ab_tensor
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy tensor to avoid breaking the training loop
            dummy_l = torch.zeros((1, self.size, self.size))
            dummy_ab = torch.zeros((2, self.size, self.size))
            return dummy_l, dummy_ab


def weights_init_normal(m):
    """Initialize weights with normal distribution"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    
    dataset = ColorizationDataset(args.data_root, transform=transform, size=args.img_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.n_workers
    )
    
    # Initialize models
    generator = Generator(in_channels=1, out_channels=2).to(device)
    discriminator = Discriminator(in_channels=3).to(device)  # L (1) + ab (2) channels
    
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    
    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_pixelwise = nn.L1Loss()
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=args.lr_decay_epoch, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=args.lr_decay_epoch, gamma=0.5)
    
    # Configure losses
    lambda_pixel = args.lambda_pixel
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        for i, (real_A, real_B) in enumerate(dataloader):
            # Move tensors to device
            real_A = real_A.to(device)  # L channel
            real_B = real_B.to(device)  # ab channels
            
            # Adversarial ground truths
            valid = torch.ones((real_A.size(0), 1, 16, 16), requires_grad=False).to(device)
            fake = torch.zeros((real_A.size(0), 1, 16, 16), requires_grad=False).to(device)
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate fake color channels
            fake_B = generator(real_A)
            
            # GAN loss
            pred_fake = discriminator(real_A, fake_B)
            loss_GAN = criterion_GAN(pred_fake, valid)
            
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)
            
            # Total generator loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel
            loss_G.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Real loss
            pred_real = discriminator(real_A, real_B)
            loss_real = criterion_GAN(pred_real, valid)
            
            # Fake loss
            pred_fake = discriminator(real_A, fake_B.detach())
            loss_fake = criterion_GAN(pred_fake, fake)
            
            # Total discriminator loss
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()
            
            # Print progress
            batches_done = epoch * len(dataloader) + i
            if i % args.report_interval == 0:
                time_elapsed = time.time() - start_time
                time_per_batch = time_elapsed / (batches_done + 1)
                eta = time_per_batch * (args.epochs * len(dataloader) - batches_done)
                
                print(
                    f"[Epoch {epoch}/{args.epochs}] "
                    f"[Batch {i}/{len(dataloader)}] "
                    f"[D loss: {loss_D.item():.4f}] "
                    f"[G loss: {loss_G.item():.4f}, pixel: {loss_pixel.item():.4f}, adv: {loss_GAN.item():.4f}] "
                    f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}"
                )
            
            # Save sample images
            if batches_done % args.sample_interval == 0:
                save_sample_images(real_A, real_B, fake_B, args.save_dir, batches_done)
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        
        # Save model checkpoints
        if (epoch + 1) % args.checkpoint_interval == 0 or epoch == args.epochs - 1:
            torch.save(generator.state_dict(), f"{args.save_dir}/generator_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"{args.save_dir}/discriminator_epoch_{epoch}.pth")
            print(f"Saved checkpoint for epoch {epoch}")
        
        epoch_time = time.time() - epoch_start_time
        print(f"Completed epoch {epoch} in {time.strftime('%H:%M:%S', time.gmtime(epoch_time))}")
    
    # Save final models
    torch.save(generator.state_dict(), f"{args.save_dir}/pix2pix_generator.pth")
    torch.save(discriminator.state_dict(), f"{args.save_dir}/pix2pix_discriminator.pth")
    
    total_training_time = time.time() - start_time
    print(f"Training completed in {time.strftime('%H:%M:%S', time.gmtime(total_training_time))}")


def save_sample_images(real_L, real_ab, fake_ab, save_dir, batches_done, max_samples=4):
    """Save sample colorization results during training"""
    # Create samples directory if it doesn't exist
    samples_dir = f"{save_dir}/samples"
    os.makedirs(samples_dir, exist_ok=True)
    
    # Convert tensors to numpy arrays
    real_L = real_L.detach().cpu().numpy()
    real_ab = real_ab.detach().cpu().numpy()
    fake_ab = fake_ab.detach().cpu().numpy()
    
    # Take up to max_samples
    max_samples = min(max_samples, real_L.shape[0])
    
    # Create figure
    fig, axs = plt.subplots(max_samples, 3, figsize=(12, 4 * max_samples))
    
    for i in range(max_samples):
        # Original L channel
        L = real_L[i][0]
        L = L * 50 + 50  # Denormalize
        
        # Ground truth ab channels
        ab_real = real_ab[i]
        ab_real = ab_real * 128  # Denormalize
        ab_real = np.transpose(ab_real, (1, 2, 0))
        
        # Generated ab channels
        ab_fake = fake_ab[i]
        ab_fake = ab_fake * 128  # Denormalize
        ab_fake = np.transpose(ab_fake, (1, 2, 0))
        
        # Combine channels to create LAB images
        original_lab = np.concatenate([L[:, :, np.newaxis], np.zeros_like(ab_real)], axis=2)
        real_lab = np.concatenate([L[:, :, np.newaxis], ab_real], axis=2)
        fake_lab = np.concatenate([L[:, :, np.newaxis], ab_fake], axis=2)
        
        # Convert LAB to RGB
        original_rgb = np.clip(lab2rgb(original_lab), 0, 1)
        real_rgb = np.clip(lab2rgb(real_lab), 0, 1)
        fake_rgb = np.clip(lab2rgb(fake_lab), 0, 1)
        
        # Plot
        if max_samples > 1:
            axs[i, 0].imshow(original_rgb)
            axs[i, 0].set_title("Grayscale")
            axs[i, 0].axis("off")
            
            axs[i, 1].imshow(real_rgb)
            axs[i, 1].set_title("Original")
            axs[i, 1].axis("off")
            
            axs[i, 2].imshow(fake_rgb)
            axs[i, 2].set_title("Generated")
            axs[i, 2].axis("off")
        else:
            axs[0].imshow(original_rgb)
            axs[0].set_title("Grayscale")
            axs[0].axis("off")
            
            axs[1].imshow(real_rgb)
            axs[1].set_title("Original")
            axs[1].axis("off")
            
            axs[2].imshow(fake_rgb)
            axs[2].set_title("Generated")
            axs[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(f"{samples_dir}/batch_{batches_done}.png")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pix2Pix GAN for Image Colorization")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset")
    parser.add_argument("--save_dir", type=str, default="model", help="Directory to save models and samples")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--lr_decay_epoch", type=int, default=100, help="Epoch to decay learning rate")
    parser.add_argument("--lambda_pixel", type=float, default=100, help="Weight for pixel-wise loss")
    parser.add_argument("--img_size", type=int, default=256, help="Size of the images")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--sample_interval", type=int, default=500, help="Interval between sampling images")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Interval between saving model checkpoints")
    parser.add_argument("--report_interval", type=int, default=100, help="Interval between printing progress")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
