from matplotlib import pyplot as plt
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import random
from PIL import Image
from PIL import ImageEnhance
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import torchvision.utils as vutils
from torchvision.utils import save_image
import argparse 
from barbar import Bar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def preprocess(image_path):

    input_image = load_image(image_path)
    input_image = np.array(input_image, dtype=np.double) / 255.0

    input_image = TF.to_pil_image(input_image)

    resize = transforms.Resize(size=(256, 256))
    input_image = resize(input_image)
    input_image = np.array(input_image)
    input_image = TF.to_tensor(input_image)
    input_image = input_image.to(device)
    input_image = input_image.float()
    input_image = input_image.unsqueeze(0)

    return input_image