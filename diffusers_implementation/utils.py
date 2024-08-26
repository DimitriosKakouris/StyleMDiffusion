from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import imageio




transform = transforms.ToTensor()

# Apply the transformation to the image


def normalize(image):
    image = np.array(image)

    image = image / 127.5 - 1
    image = transform(image)
    image = torch.tensor(image).unsqueeze(0)
    # print(f"Image shape: {image.shape}")
    print(f"Image shape: {image.shape} and type: {type(image)}")
    image = F.interpolate(image, size=512, mode='bilinear', align_corners=False)

    # print(f"Image shape: {image.shape} and type: {type(image)}")
    # image_data = np.transpose(image.cpu().numpy(), (1,2,3,0))
    
    # image_data = denormalize(image)
    # image_data = np.squeeze(image_data)

    # plt.imshow(image_data)
    # plt.savefig('normalized_image.png')

    # print(f"Image shape: {image.shape} and type: {type(image)}")
    return image

def denormalize(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    return (image * 255).round().astype("uint8")

def resize(image, size):
    image = image.resize(size, Image.LANCZOS)
    return image

def color_cluster(image, color_mapping=None):
    """
    Image is numpy array with distinct integer index
    """
    if color_mapping is None:
        color_mapping = [
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (255, 255, 0),   # Yellow
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Cyan
            (255, 128, 0),   # Orange
            (128, 0, 255),   # Purple
            (128, 128, 128), # Gray
            (128, 128, 0),   # Olive
            (0, 128, 128),   # Teal
            (128, 0, 128),   # Maroon
            (0, 128, 0),     # Lime
            (128, 0, 0),     # Navy
            (0, 0, 128),     # Fuchsia
            (0, 0, 0)        # Black
        ]
        
    new_image = np.zeros([*image.shape, 3], dtype=np.uint8)
    
    H, W = image.shape
    for h in range(H):
        for w in range(W):
            new_image[h, w] = color_mapping[image[h, w]]
            
    return new_image
            
# OPs for saving image from raw data
def save_image(image, filename):
    """
    Image should be in range (0, 255) and numpy array
    """
    image = Image.fromarray(image)
    image.save(filename)
    
    
def save_cluster(image, filename, color_mapping=None, size=None):
    """
    Image is numpy array with distinct integer index
    """
    new_image = color_cluster(image, color_mapping)
    
    image = Image.fromarray(new_image)
    
    if size is not None:
        image = image.resize(size, Image.LANCZOS)
    image.save(filename)
    
    

def array_to_gif(image_array, output_path, duration=0.1):
    # Convert the image array to uint8 format
    if image_array.dtype == np.float32:
        image_array = (255 * np.clip(image_array, 0, 1)).astype(np.uint8)

    # Create a list to store individual frames
    frames = []
    for image in image_array:
        frames.append(image)

    # Save the frames as a GIF
    imageio.mimsave(output_path, frames, duration=duration)
    
    