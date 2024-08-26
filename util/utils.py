import torch
import torch.nn as nn
from torchvision.transforms import Resize, ToTensor
from CLIP import clip


def resize_image(image):
    resize = Resize((512, 512))
    image = resize(image)
    image = ToTensor()(image)
    return image
