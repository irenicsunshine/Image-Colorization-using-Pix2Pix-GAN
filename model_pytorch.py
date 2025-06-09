"""
Image Colorization Model using PyTorch

This module contains a simple U-Net style neural network for image colorization.
We'll use this as a replacement for the outdated Caffe model.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from skimage.color import rgb2lab, lab2rgb


class UnetBlock(nn.Module):
    """U-Net building block with skip connections"""
    def __init__(self, outer_channels, inner_channels, input_channels=None,
                 submodule=None, outermost=False, innermost=False, 
                 norm_layer=nn.BatchNorm2d, dropout=0.0):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        
        if input_channels is None:
            input_channels = outer_channels
            
        downconv = nn.Conv2d(input_channels, inner_channels, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_channels)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_channels)
        
        if outermost:
            upconv = nn.ConvTranspose2d(inner_channels * 2, outer_channels,
                                        kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_channels, outer_channels,
                                       kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_channels * 2, outer_channels,
                                       kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            
            if dropout > 0:
                model = down + [submodule] + up + [nn.Dropout(dropout)]
            else:
                model = down + [submodule] + up
                
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class ColorGenerator(nn.Module):
    """U-Net based generator for image colorization"""
    def __init__(self, input_channels=1, output_channels=2, num_downs=8):
        super(ColorGenerator, self).__init__()
        
        # Construct a U-Net structure
        unet_block = UnetBlock(512, 512, input_channels=None, submodule=None, innermost=True)
        
        for i in range(num_downs - 5):
            unet_block = UnetBlock(512, 512, input_channels=None, submodule=unet_block, dropout=0.5)
            
        unet_block = UnetBlock(256, 512, input_channels=None, submodule=unet_block)
        unet_block = UnetBlock(128, 256, input_channels=None, submodule=unet_block)
        unet_block = UnetBlock(64, 128, input_channels=None, submodule=unet_block)
        self.model = UnetBlock(output_channels, 64, input_channels=input_channels, 
                               submodule=unet_block, outermost=True)
        
    def forward(self, x):
        return self.model(x)


class ColorizationModel:
    """Wrapper class for the colorization model"""
    def __init__(self, model_path=None, device=None):
        # Determine the device to use
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create the model
        self.model = ColorGenerator()
        
        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print("Initializing model with random weights")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image transformations
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])
        
    def preprocess(self, img):
        """Convert RGB image to L channel"""
        img_lab = rgb2lab(np.array(img) / 255.0)
        img_l = img_lab[:, :, 0]
        # Add batch and channel dimensions
        tensor_l = torch.from_numpy(img_l).unsqueeze(0).unsqueeze(0).float()
        # Normalize to [-1, 1]
        tensor_l = (tensor_l - 50) / 50
        return tensor_l
    
    def postprocess(self, img_l, output_ab):
        """Combine L channel with predicted AB channels"""
        img_l = img_l.squeeze().cpu().numpy()
        # Unnormalize lightness
        img_l = img_l * 50 + 50
        # Get the predicted AB channels
        output_ab = output_ab.squeeze().cpu().numpy()
        # Transpose from (2, H, W) to (H, W, 2)
        output_ab = np.transpose(output_ab, (1, 2, 0))
        # Unnormalize AB channels
        output_ab = output_ab * 128
        # Combine L and AB channels
        result_lab = np.concatenate([img_l[:, :, np.newaxis], output_ab], axis=2)
        # Convert from LAB to RGB
        result_rgb = lab2rgb(result_lab)
        # Convert to uint8
        result_rgb = (result_rgb * 255).astype(np.uint8)
        return result_rgb
    
    def colorize(self, img):
        """Colorize a single image"""
        # Set model to evaluation mode
        self.model.eval()
        
        # Preprocess the input image
        l_channel = self.preprocess(img).to(self.device)
        
        with torch.no_grad():
            # Forward pass to get AB channels
            ab_channels = self.model(l_channel)
            # Convert to proper range for visualization
            ab_channels = ab_channels * 128
            
        # Combine L and predicted AB channels
        colorized = self.postprocess(l_channel, ab_channels)
        return colorized


# Initialize a basic model with random weights for testing
def get_pretrained_model():
    """Get a pretrained model or one initialized with random weights"""
    # In a real application, we would either:
    # 1. Download a pre-trained model from a reliable source
    # 2. Train our own model and save it
    # For demonstration, we'll use a randomly initialized model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/colorization_model.pth")
    return ColorizationModel(model_path=model_path if os.path.exists(model_path) else None)


if __name__ == "__main__":
    # Simple test code
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Load test image
    img = Image.open("test.jpg").convert("RGB")
    
    # Get model and colorize
    model = get_pretrained_model()
    colorized = model.colorize(img)
    
    # Show results
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    axes[1].imshow(colorized)
    axes[1].set_title("Colorized")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()
