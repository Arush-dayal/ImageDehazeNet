# Using a custom GAN architecture based on the pix2pixHD model for image de-hazing task.
The model has been trained to generate clear images given input hazy images, or a dehazing task. A GAN model has been used for this purpose. A custom architecture has been implemented for the GAN model which is based on the pix2pixHD architecture. 

The pix2pixHD architcture and its variations have been described in the following research paper:
https://openaccess.thecvf.com/content_CVPR_2019/papers/Qu_Enhanced_Pix2pix_Dehazing_Network_CVPR_2019_paper.pdf
 
This particular GAN model uses a local generator and global generator as well as two discrminators that work at different resolutions. 

### Generator:
Each of the generators have a similar U-net architecture with skip connections implemented in between the downsampling and upsampling layers. The downsampling and upsampling layers consist of convolutional blocks.

The global generator first instantiates the local generator which processes the input image at half the resolution and then the output image is processed by the global generator to generate the high resolution output which will be passed into the discriminators. The presence of a low resolution local generator allows the model to learn to generate certain high level features of the image, such as the object boundaries, which is important in an image de-hazing task.


### Discriminator:
The discriminators involve convolutional layers followed by a batchnorm layer. Two discriminators of different resolutions are implemented in order to help the model learn high level structural details in the image as well as certain important pixel level details.


The loss function for the model is a combination of adverserial loss as well as the reconstruction loss (MSE).




## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) in the terminal to install the following packages and libraries:

```bash
py -m pip install matplotlib torch torch.nn numpy torchvision torh.utils PIL scikit-image argparse barbar
```

## Usage

->Make sure test.py imports the preprocess() function from preprocess.py.

->Set variable checkpoint_dir as path to where final_model.pth checkoint is stored for loading the final trained model params.

->After loading state, run model on test dataset.
