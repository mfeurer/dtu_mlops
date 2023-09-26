import torch
import numpy as np
from torchvision import transforms
import os


from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image).float()

        return image, label



def mnist():
    # List of file names
    file_names = ['train_0.npz', 'train_1.npz', 'train_2.npz', 'train_3.npz', 'train_4.npz']
    path = "/Users/LeMarx/Documents/01_Projects/MLOps/dtu_mlops/data/corruptmnist/"
    # Initialize empty lists to store images and labels
    images_list = []
    labels_list = []

    #arr_comb = np.concatenate((arr,arr2), axis = 0)

    # Load and merge data from each .npz file
    for file_name in file_names:
        data = np.load(os.path.join(path, file_name))
        images = data['images']  
        labels = data['labels']
        
        # Append the data to the lists
        images_list.append(images)
        labels_list.append(labels)

    # Concatenate the arrays from all files


    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

    trainset = CustomDataset(images_list,labels_list, transform= transform)
    #testset = CustomDataset(np.load(os.path.join(path,"test.npz"))["images"],os.path.join(path,"test.npz")["labels"],transform=transform)


    trainloader = DataLoader(trainset, 32, shuffle = True)
    #testloader = DataLoader(testset,32,True)

    return trainloader,"testloader"


