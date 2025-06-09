"""
Pix2Pix GAN Model for Image Colorization

This module contains an implementation of Pix2Pix GAN for image colorization.
Architecture based on "Image-to-Image Translation with Conditional Adversarial Networks"
by Isola et al. (https://arxiv.org/pdf/1611.07004.pdf)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb


class UNetDown(nn.Module):
    """U-Net downsampling block"""
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """U-Net upsampling block"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class Generator(nn.Module):
    """Pix2Pix Generator with U-Net architecture"""
    def __init__(self, in_channels=1, out_channels=2):
        super(Generator, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # U-Net architecture with skip connections
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        
        return self.final(u7)


class Discriminator(nn.Module):
    """Pix2Pix Discriminator - PatchGAN"""
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Input: [grayscale image, generated ab channels or real ab channels]
        # for the conditional GAN structure, input is the concatenated channel
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate input and condition image by channels
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class Pix2PixGAN:
    """Wrapper class for the Pix2Pix GAN model for colorization"""
    def __init__(self, generator_path=None, device=None):
        # Set device for model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize generator (the only component needed for inference)
        self.generator = Generator(in_channels=1, out_channels=2)
        self.generator = self.generator.to(self.device)
        
        # Load pre-trained weights for generator if provided
        if generator_path and os.path.exists(generator_path):
            self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
            print(f"Loaded generator from {generator_path}")
        else:
            print("Initializing generator with random weights")
        
        # Set model to evaluation mode for inference
        self.generator.eval()
        
        # Image preprocessing transforms
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def preprocess(self, img):
        """
        Preprocess RGB image to L channel
        Returns:
            - L channel tensor normalized to [-1, 1]
            - original image dimensions
        """
        # Store original dimensions for resizing back
        original_size = img.size
        
        # Convert to LAB color space
        img_lab = rgb2lab(np.array(img) / 255.0)
        
        # Extract L channel
        img_l = img_lab[:, :, 0]
        
        # Normalize L channel to [-1, 1]
        img_l = (img_l - 50) / 50
        
        # Convert to tensor with batch dimension
        tensor_l = torch.from_numpy(img_l).float().unsqueeze(0).unsqueeze(0)
        
        # Resize to 256x256 for the model
        tensor_l = F.interpolate(tensor_l, size=(256, 256), mode='bilinear', align_corners=True)
        
        return tensor_l, original_size

    def postprocess(self, l_tensor, ab_tensor, original_size):
        """
        Convert generated ab channels and original L channel back to RGB
        """
        # Resize ab channels back to original dimensions
        ab_resized = F.interpolate(ab_tensor, size=(original_size[1], original_size[0]), 
                                 mode='bilinear', align_corners=True)
        
        # Resize L channel back to original dimensions
        l_resized = F.interpolate(l_tensor, size=(original_size[1], original_size[0]),
                                mode='bilinear', align_corners=True)
        
        # Convert tensors to numpy
        l_np = l_resized.squeeze().cpu().numpy()
        ab_np = ab_resized.squeeze().cpu().numpy()
        
        # Denormalize L channel from [-1, 1] to [0, 100]
        l_np = l_np * 50 + 50
        
        # Denormalize ab channels from [-1, 1] to [-128, 128]
        ab_np = ab_np * 128
        
        # Reshape ab channels from [2, H, W] to [H, W, 2]
        ab_np = np.transpose(ab_np, (1, 2, 0))
        
        # Combine L and ab channels
        lab_image = np.concatenate([l_np[:, :, np.newaxis], ab_np], axis=2)
        
        # Convert LAB back to RGB
        rgb_image = lab2rgb(lab_image)
        
        # Convert to uint8 format
        rgb_image = (rgb_image * 255).astype(np.uint8)
        
        return rgb_image

    def colorize(self, img):
        """
        Colorize a grayscale image using the Pix2Pix GAN
        Args:
            img: PIL Image in RGB format
        Returns:
            Colorized image as numpy array
        """
        self.generator.eval()  # Set to evaluation mode
        
        with torch.no_grad():  # No need to track gradients during inference
            # Preprocess input image to get L channel and original dimensions
            l_channel, original_size = self.preprocess(img)
            
            # Move to device
            l_channel = l_channel.to(self.device)
            
            # Generate ab channels
            ab_channels = self.generator(l_channel)
            
            # Postprocess to get the colorized image
            colorized = self.postprocess(l_channel, ab_channels, original_size)
            
        return colorized


def get_pretrained_model():
    """Get a pretrained Pix2Pix GAN model for colorization"""
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    generator_path = os.path.join(model_dir, "pix2pix_generator.pth")
    
    # Create the model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    return Pix2PixGAN(generator_path=generator_path if os.path.exists(generator_path) else None)


# Training function for the Pix2Pix GAN (not used during inference)
def train_pix2pix(dataloader, epochs=100, sample_interval=100, save_path='model'):
    """
    Train the Pix2Pix GAN model
    
    Note: This is a simplified training function. For actual training,
    a proper dataset class and training loop would be needed.
    """
    import torch.optim as optim
    from torch.utils.data import DataLoader
    
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = Discriminator(in_channels=3).to(device)  # 1 (L) + 2 (ab) = 3 channels
    
    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_pixelwise = nn.L1Loss()
    
    # Loss weights
    lambda_pixel = 100
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Training loop
    for epoch in range(epochs):
        for i, (real_A, real_B) in enumerate(dataloader):
            # Move tensors to device
            real_A = real_A.to(device)  # L channel
            real_B = real_B.to(device)  # ab channels
            
            # Adversarial ground truths
            valid = torch.ones((real_A.size(0), 1, 16, 16), requires_grad=False).to(device)
            fake = torch.zeros((real_A.size(0), 1, 16, 16), requires_grad=False).to(device)
            
            # ------------------
            #  Train Generator
            # ------------------
            optimizer_G.zero_grad()
            
            # Generate fake ab channels
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
            
            # --------------------
            #  Train Discriminator
            # --------------------
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
            if i % 100 == 0:
                print(
                    f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {loss_D.item()}] [G loss: {loss_G.item()}, pixel: {loss_pixel.item()}, adv: {loss_GAN.item()}]"
                )
        
        # Save model checkpoints
        if epoch % 10 == 0:
            torch.save(generator.state_dict(), f"{save_path}/pix2pix_generator_epoch_{epoch}.pth")
    
    # Save final model
    torch.save(generator.state_dict(), f"{save_path}/pix2pix_generator.pth")
    
    return generator, discriminator


if __name__ == "__main__":
    # Simple test
    import matplotlib.pyplot as plt
    
    # Create a model
    model = get_pretrained_model()
    
    # Test with a sample grayscale image
    test_img = Image.open("test.jpg").convert('RGB')
    
    # Colorize
    colorized = model.colorize(test_img)
    
    # Display
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(test_img)
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(colorized)
    plt.title("Colorized")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
