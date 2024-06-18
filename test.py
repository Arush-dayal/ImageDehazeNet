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

from model import Local_Generator
from model import Generator
from preprocess import preprocess

# TEST_DATASET_HAZY_PATH = '/home/user/Desktop/test/hazy'
# TEST_DATASET_OUTPUT_PATH = '/home/user/Desktop/test/dehazed_output'

# MY try:
TEST_DATASET_HAZY_PATH = 'C:/Users/arush/Downloads/test_try/hazy'
TEST_DATASET_OUTPUT_PATH = 'C:/Users/arush/Downloads/test_try/results'

input_images = os.listdir(TEST_DATASET_HAZY_PATH)
num = len(input_images)
output_images = []

for i in range(num):
    output_images.append(os.path.join(TEST_DATASET_OUTPUT_PATH, input_images[i]))
    input_images[i] = os.path.join(TEST_DATASET_HAZY_PATH, input_images[i])

'''
Write the code here to load your model
'''
checkpoint_path = './final_model.pth' #Vary accordingly.
checkpoint = torch.load(checkpoint_path)

local = Local_Generator()
model = Generator(local)
model.local_gen.load_state_dict(checkpoint['local_generator_state_dict'])
model.local_gen.eval()
model.load_state_dict(checkpoint['generator_state_dict'])
model.eval()

for i in range(num):
    
    #calling preprocessing fn before passing input into model for testing.
    input_image = preprocess(input_images[i])

    #now save the dehazed image at the path indicated by output_images[i]
    dehazed_image = model(input_image)

    dehazed_image = dehazed_image.clamp(0, 1)
    dehazed_image_numpy = dehazed_image[0].permute(1, 2, 0).detach().cpu().numpy()
    # Save the NumPy array as an image file
    plt.imsave(output_images[i], dehazed_image_numpy)

    # save_image(dehazed_image, output_images[i])
    