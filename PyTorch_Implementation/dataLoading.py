# coding custom dataloader


# import needed libraries
import torch
import os
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#plt.ion()   # interactive mode

print("all libraries are imported successfully...")

csv_path = r"F:\Meine Programme\OpenAI_RaceGame\Training_Data\training_v2\training\labels.csv"


# create a dataset class
# torch.utils.data.Dataset is an abstract clas representing a dataset

class SteeringCommands(Dataset): 
    """ Steering Commands dataset """

    def __init__(self, csv_file, root_dir, transform=None): 
        """
            Args:
                csv_file (string): Path to the csv file with annotations.
                root_dir (string): Directory with all the images.
                transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transform

    def __len__(self): 
        return len(self.label_file)

    def __getitem__(self, idx): 
        if torch.is_tensor(idx): 
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.label_file.iloc[idx,11])
        image = io.imread(img_name)
        image = image[:,:,0:2]      #remove blue color channel from image
        commands = self.label_file.iloc[idx, 8:11]
        commands = np.asarray(commands)
        if commands[0] == 1.0:           #append 0/1 depending on turning key value -> defines input for right key
            commands = np.concatenate(([1.0],commands))
        else:
            commands = np.concatenate(([0.0],commands))
        if commands[1] == -1.0:           #append 0/1 depending on turning key value -> defines input for left key
            commands =  np.concatenate(([1.0],commands))
        else:
            commands = np.concatenate(([0.0],commands))
        commands = np.delete(commands, 2)
        image = image.astype('float')
        input_data = self.label_file.iloc[idx, 0:8]
        input_data = np.asarray(input_data)
        commands = commands.astype('float').reshape(-1)
        input_data = input_data.astype('float').reshape(-1)


        if self.transforms: 
            image = self.transforms(image)
            
        commands = torch.from_numpy(commands)
        input_data = torch.from_numpy(input_data)
        sample = {'image': image, 'input_data' : input_data,'commands': commands}
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        return image

# instantiate this class
convert_data = ToTensor()       #initialize convert_data class which is used to transpose image in steeringcommands class


#____________________________________________________________________________________________

# data transformations
# transformations used at training time is referred to data augmentation
# three types of transforms: 
#   - Rescale (not important for us since we have all images with same scale
#   - RandomCrop (crop image randomly -> data augmentation)
#   - ToTensor (convert numpy images to torch images swapping axes)
"""
tsfm = Transform(params)
transformed_sample = tsfm(sample)

class RandomCrop(Object): 
    Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.

class ToTensor(object):
    Convert ndarrays in sample to Tensors

"""

# iterate through the dataset

"""
transformed_dataset = FaceLandmarksDataset(csv_file='faces/face_landmarks.csv',
                                           root_dir='faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))
"""

transformed_dataset = SteeringCommands(csv_file=csv_path, root_dir=r"F:\Meine Programme\OpenAI_RaceGame\Training_Data\training_v2\training", transform=convert_data)
#transformed dataset receives csf_file path and image directory path as input, as well as the transform operation defined in the ToTensor() class, which is called for each getitem call





# _____________________________________________________________________________________

# the dataloader: 
    


# number of subprocesses to use for data loading
num_workers = 0

# samples per batch to load
batch_size = 4

# % of training set to use as validation
valid_size = 0.2

"""
dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

for i_batch, sample_batched in enumerate(dataloader):
    #print(sample_batched['commands'])
    pass
"""

# prepare dataloaders
#train_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#valid_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)