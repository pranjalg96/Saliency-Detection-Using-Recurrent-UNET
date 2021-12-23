# Saliency Detection using Recurrent U-NET
Explores a recurrent architecture of the U-NET which helps incorporate saliency priors for more accurate prediction. Saliency priors are obtained using
Gaussian Pyramids and Gabor filters. For more details, please look at 'ECE6258_Term_Paper.pdf'. There are many models presented in the paper, but this repo 
contains code only for the Saliency Heuristics recurrent U-NET model.

## Dataset
The dataset used is PASCAL VOC 2010. For your convenience, I have done the reading of both the images and ground truth saliency maps and compressed them
into two .npy files that you just need to load in the notebook (make sure the path to the two files is correct in the notebook, I used google colab). The
two files are further described below:

X_UNET_recurr_prior_map.npy: This file contains all images from the dataset in a numpy array of the form (num_images, 224, 224, 4), where the first 3
channels are the RGB channels of the image, and the 4th channel is the saliency prior of the image, obtained using the heuristics discussed in the paper.
google drive link: https://drive.google.com/file/d/1EAu7lLK9VMf__FwvpJF_7U3DzVt0MTwB/view?usp=sharing

y_UNET_recurr_prior_map.npy: This file contains the ground truth images in a numpy array of the form (num_images, T, 224, 224, 1), where
T is the number of time steps of the recurrent U-NET. Here, we use T = 3. 
google drive link: https://drive.google.com/file/d/1eGxaaIJsNZox3DjTIYY4IvkOm_7Kt-6Y/view?usp=sharing

I know that compressing all this data into the two files will not be helpful if you want to try the model on a different dataset. Please contact me if you have
any troubles in training the model on a different dataset. 

## Notebook
recurr_UNET_sal_priors.ipynb: This is used to read the above two files. It also performs splitting into train and validation sets. Then
it builds the recurrent U-NET model and trains it. Finally, saliency map predictions on both the training and validation sets are visualized. 

## Contact
Please email me if you have any problems getting it to work.
email: pranjalg96@gmail.com