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
from skimage.filters import gaussian
from skimage.filters import unsharp_mask
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import torchvision.utils as vutils
import argparse 
from barbar import Bar

#Importing model components from model.py file
from model import Discriminator1
from model import Discriminator2
from model import Local_Generator
from model import Generator

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''Loading of the training dataset'''

def load_image(image_path):
    return Image.open(image_path).convert("RGB")


def load_and_augment_dataset(dataset_path, num_augmentations=1): #Can alter number of augmentations to be performed on data here
    augmented_pairs = []

    # Path to the input and output subfolders
    input_folder_path = os.path.join(dataset_path, 'train', 'hazy')
    output_folder_path = os.path.join(dataset_path, 'train', 'GT')

    # List all image files in the input folder
    image_files = os.listdir(input_folder_path)
    image_files.sort()

    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # Iterate over image files
    for image_file in image_files:
        # Construct the full path for input and output images
        input_image_path = os.path.join(input_folder_path, image_file)
        output_image_path = os.path.join(output_folder_path, image_file)

        # Load input and output images
        input_image = load_image(input_image_path)
        output_image = load_image(output_image_path)

        # Resize input image to match the size of the output image
        input_image = input_image.resize(output_image.size, resample=Image.Resampling.BILINEAR)

        # Convert input and output images to numpy arrays
        input_image = np.array(input_image, dtype=np.double) / 255.0  # Normalize to range [0, 1]
        output_image = np.array(output_image, dtype=np.float64) / 255.0

        # Augment images and append to augmented_pairs list
        for _ in range(num_augmentations):
            input_image_augmented, output_image_augmented = transform_images(input_image, output_image)
            augmented_pairs.append((input_image_augmented, output_image_augmented))

    return augmented_pairs


''' Augmentation of the dataset'''

def transform_images(input_image, output_image):

    # Convert numpy arrays to PIL images
    input_image = TF.to_pil_image(input_image)
    output_image = TF.to_pil_image(output_image)

    resize = transforms.Resize(size=(256, 256))
    input_image = resize(input_image)
    output_image = resize(output_image)

    #Randomly horizontally flips the pair of images
    if random.random() > 0.7:
        input_image = TF.hflip(input_image)
        output_image = TF.hflip(output_image)

    #Randomly vertically flips the pair of images
    if random.random() > 0.7:
        input_image = TF.vflip(input_image)
        output_image = TF.vflip(output_image)

    #Randomly changes the brightness of the pair of images
    brightness_factor = random.uniform(0.8, 1.2)
    input_image = ImageEnhance.Brightness(input_image).enhance(brightness_factor)
    output_image = ImageEnhance.Brightness(output_image).enhance(brightness_factor)

    #Randomly changes the contrast of the pair of images
    contrast_factor = random.uniform(0.8, 1.2)
    input_image = ImageEnhance.Contrast(input_image).enhance(contrast_factor)
    output_image = ImageEnhance.Contrast(output_image).enhance(contrast_factor)

    #Randomly changes the saturation of the pair of images
    saturation_factor = random.uniform(0.8, 1.2)
    input_image = ImageEnhance.Color(input_image).enhance(saturation_factor)
    output_image = ImageEnhance.Color(output_image).enhance(saturation_factor)

    #Randomly changes the hue of the pair of images
    hue_factor = random.uniform(0.8, 1.2)
    input_image = ImageEnhance.Sharpness(input_image).enhance(hue_factor)
    output_image = ImageEnhance.Sharpness(output_image).enhance(hue_factor)

    input_image = np.array(input_image)
    output_image = np.array(output_image)

    #Randomly adds gaussian blur to the pair of images
    if random.random() < 0.25:
        sigma_param = random.uniform(0.1, 0.5)
        input_image = gaussian(input_image, sigma=sigma_param)
        output_image = gaussian(output_image, sigma=sigma_param)
    
    #Randomly adds unsharp mask to the pair of images
    if random.random() < 0.25:
        radius_param = random.uniform(0, 2)
        amount_param = random.uniform(0.2, 1)
        input_image = unsharp_mask(input_image, radius=radius_param, amount=amount_param)
        output_image = unsharp_mask(output_image, radius=radius_param, amount=amount_param)
    
    input_image = TF.to_tensor(input_image)
    output_image = TF.to_tensor(output_image)

    return input_image, output_image

dataset_path = "C:/Users/arush/Downloads/final_dataset (train+val)" #Update path to the folder containing the training and val dataset
augmented_pairs = load_and_augment_dataset(dataset_path)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_image, output_image = self.pairs[idx]
        return input_image, output_image

custom_dataset = CustomDataset(augmented_pairs)
batch_size = 50
data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)


'''Initializing weights to be updated'''

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        if m.bias is not None:
            m.bias.data.fill_(0)

torch.autograd.set_detect_anomaly(True)


'''Trainer Class for training the model'''
class Trainer:
    def __init__(self, args, data, device, checkpoint_dir):
        self.args = args
        self.train_loader = data
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    '''Visualizing the input, target and generated images'''
    def visualize(self, input_image, output_image, generated_image):

        fig, axes = plt.subplots(1, 3)
            
        # Convert input image to numpy array for visualization
        input_image_numpy = input_image[0].permute(1, 2, 0).cpu().numpy()
        axes[0].imshow(input_image_numpy)
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        # Convert output image to numpy array for visualization
        output_image_numpy = output_image[0].permute(1, 2, 0).cpu().numpy()
        axes[1].imshow(output_image_numpy)
        axes[1].set_title('Output Image')
        axes[1].axis('off')

        # Clip and normalize generated image pixel values to range [0, 1]
        # generated_image_clipped = torch.clamp(generated_image, min=0, max=1)
        generated_image = generated_image.clamp(0, 1)
        generated_image_numpy = generated_image[0].permute(1, 2, 0).detach().cpu().numpy()
        # generated_image_numpy = generated_image[0].permute(1, 2, 0).detach().cpu().numpy()
        axes[2].imshow(generated_image_numpy)
        axes[2].set_title('Generated Image')
        axes[2].axis('off')

        plt.show()


    '''Defining the training loop'''

    def train(self):

        checkpoint_path = './final_model.pth' 
        checkpoint = torch.load(checkpoint_path)

        self.G1 = Local_Generator().to(self.device)
        #self.G1.apply(weights_init_normal)
        self.G = Generator(self.G1).to(self.device)
        self.D = Discriminator1().to(self.device)
        self.D2 = Discriminator2().to(self.device)

        self.G.load_state_dict(checkpoint['generator_state_dict'])
        self.G.local_gen.load_state_dict(checkpoint['local_generator_state_dict'])
        self.D.load_state_dict(checkpoint['discriminator1_state_dict'])
        self.D2.load_state_dict(checkpoint['discriminator2_state_dict'])

        # self.G.apply(weights_init_normal)
        # self.D.apply(weights_init_normal)
        # self.D2.apply(weights_init_normal)

        optimizer_g = optim.Adam(list(self.G.parameters()) +
                                      list(self.G1.parameters()), lr=self.args.lr_adam, weight_decay=0.001)
        optimizer_d = optim.Adam(list(self.D.parameters()) +
                                      list(self.D2.parameters()), lr=self.args.lr_adam, weight_decay=0.001)
        
        criterion = nn.BCELoss()
        fidelity_loss = nn.MSELoss()

        for epoch in range(self.args.num_epochs+1):
            g_losses = 0
            d_losses = 0

            for input_image, output_image in Bar(self.train_loader):

                input_image = input_image.to(self.device)
                output_image = output_image.to(self.device)

                input_image = input_image.float()
                output_image = output_image.float()

                optimizer_g.zero_grad()
                optimizer_d.zero_grad()

                self.D.zero_grad()
                self.D2.zero_grad()
                # Train discriminator with real images
                out_true1 = self.D(input_image, output_image)
                y_true1 = torch.ones_like(out_true1)
                d1_real_loss = criterion(out_true1, y_true1)

                out_true2 = self.D2(input_image, output_image)
                y_true2 = torch.ones_like(out_true2)
                d2_real_loss = criterion(out_true2, y_true2)

                # Generate fake images
                x_generated = self.G(input_image)

                out_fake1 = self.D(input_image, x_generated)
                y_fake1 = torch.ones_like(out_fake1)
                d1_fake_loss = criterion(out_fake1, y_fake1)

                out_fake2 = self.D2(input_image, x_generated)
                y_fake2 = torch.ones_like(out_fake2)
                d2_fake_loss = criterion(out_fake2, y_fake2)
            
                d1_loss = d1_real_loss + d1_fake_loss
                d2_loss = d2_real_loss + d2_fake_loss
                d_loss = d1_loss + d2_loss
                d_loss.backward(retain_graph=True)
                optimizer_d.step()
                
                # Update generator
                self.G.zero_grad()
                self.G.local_gen.zero_grad()
                #Train generator to fool discriminator
                out_fake1 = self.D(input_image, x_generated)
                out_fake2 = self.D2(input_image, x_generated)
                
                #Fidelity loss between generated and target image calculated
                fid_loss = fidelity_loss(x_generated, output_image)
                #Adverserial loss for generator calculated with respect to both scaled Discriminators.
                g_adv1 = criterion(out_fake1, y_true1)
                g_adv2 = criterion(out_fake2, y_true2)
                g_adv = g_adv1 + g_adv2
                g_loss = g_adv + fid_loss

                g_loss.backward()
                optimizer_g.step()
                
                #Computing gradients and backpropagate           
                d_losses += d_loss.item()
                g_losses += g_loss.item()

            print("Training... Epoch: {}, Discriminator Loss: {:.3f}, Generator Loss: {:.3f}".format(
                epoch, d_losses/len(self.train_loader), g_losses/len(self.train_loader)
            ))

            # with torch.no_grad():
            #     x_ge = self.G(input_image)
            #     self.visualize(input_image, output_image, x_ge)

            if self.args.save_checkpoint:
                self.save_checkpoint(epoch)
        
        self.save_checkpoint('final')
    
    def save_checkpoint(self, epoch):
        if epoch == 'final':
            checkpoint_path = os.path.join(self.checkpoint_dir, 'final_model.pth')
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+19}.pth')

        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.G.state_dict(),
            'local_generator_state_dict': self.G1.state_dict(),
            'discriminator1_state_dict': self.D.state_dict(),  
            'discriminator2_state_dict': self.D2.state_dict(),    
        }, checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="number of epochs")
    parser.add_argument('--lr_adam', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--lr_rmsprop', type=float, default=1e-4,
                        help='learning rate RMSprop.')
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size")
    parser.add_argument('--save_checkpoint', type=bool, default=True,
                        help='If checkpoint to be saved')
    #parsing arguments.
    args = parser.parse_args() 

    #check if cuda is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = "C:/Users/arush/Downloads/final_dataset (train+val)" #Update path to folder containing the dataset
    augmented_pairs = load_and_augment_dataset(dataset_path)
    custom_dataset = CustomDataset(augmented_pairs)
    batch_size = 64
    data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
    checkpoint_dir = '.'

    model = Trainer(args, data_loader, device, checkpoint_dir)
    model.train()

