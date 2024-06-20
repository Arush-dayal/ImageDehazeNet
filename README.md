##Custom Pix2PixHD architecture for image dehazing.



Dependencies:

Create virtual environment and execute the following command:
py -m pip install {library} to install the following libraries:
matplotlib
torch
torch.nn
numpy
torchvision
torh.utils
PIL
scikit-image
argparse
barbar


Executing program:

1) model.py (Training code):
->Required data preprocessing and augmentations for training code are all present within model.py itself.

->Training file is given as model.py. In order to run the program, set dataset_path variable (2 declarations) as absolute path to the complete dataset folder -final_dataset (train+val)-, containing the train and val subfolders.

->The model checkpoints after each epoch are saved in the current working directory by default. This can be changed by setting checkpoint_dir variable to path of desired folder. 

->Hyperparamters (batch size, number of epochs and learning rate.) can be varied accordingly in __main__ , in the parser.add_argument fields at the end of the program. The hyperparamaters are created within argparse object and passed as arguments into the trainer class.

2) test.py (Testing code):
->Make sure test.py imports the preprocess() function from preprocess.py.

->Set variable checkpoint_dir as path to where final_model.pth checkoint is stored for loading the final trained model params.

->After loading state, run model on test dataset.

3) The visualizations folder in current directory contain the visualizations after each epoch obtained during the training of our model.

Acknowledgments:

Certain code snippets have been taken from the following tutorial for certain components of the model:
https://www.tensorflow.org/tutorials/generative/pix2pix

The paper referred to for the architetcure is:
https://openaccess.thecvf.com/content_CVPR_2019/papers/Qu_Enhanced_Pix2pix_Dehazing_Network_CVPR_2019_paper.pdf
