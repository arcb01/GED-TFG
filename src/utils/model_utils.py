from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pickle



class ImageDataset(Dataset):
    """
    Dataset class for the images
    Atributes:
        root_dir: path to the images
        imgs_list: list of images
        transform: data transformations
        multiclass: if True, the dataset has multiple classes
    """

    def __init__(self, root_dir, imgs_list, transform=None, multiclass=False):

        self.root_dir = root_dir
        self.imgs_list = imgs_list
        self.transform = transform
        self.multiclass = multiclass
            
        
    def get_labels(self):
        if self.multiclass:
            return np.where(self.imgs_list[:, 1::] == 1)[1]
        else:
            return self.imgs_list[:, 1].astype('int64')

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.imgs_list[idx][0])
    
        if self.multiclass:
            labels = self.imgs_list[idx][1::] # [x, y, z]  
            label = int(np.where(labels == 1)[0][0]) # idx
        else:
            label = self.imgs_list[idx][1]

        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)
        
        return image, label
                



def initialize_model(num_classes, model_name):
    """
    Initialize the model
    :param num_classes: number of classes in the dataset
    :param model_name: name of the model architecture
    """
    
    if str(model_name) == "vgg16" or str(model_name) == "convnext":
        
        if str(model_name) == "vgg16":
            model = models.vgg16(weights='IMAGENET1K_V1')
        else:
            model = models.convnext_tiny(weights='IMAGENET1K_V1')
                                         
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
                                         
    elif str(model_name) == "resnet":
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(512, num_classes)
        
    input_size = 224
        
    return model, input_size


def get_transform_normalize_values(loader):
    """
    Get the mean and std values for the dataset
    :param loader: data loader for the dataset
    """

    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0) 
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)

    return mean, std


def get_transforms(norm_path, input_size, train_dir, mc_train_data, multiclass):
    """
    Get the data transforms
    :param norm_path: path to the normalization values
    :param input_size: resize size for the images
    :param train_dir: path to the training images folder
    :param mc_train_data: training data
    :param multiclass: whether the dataset is multiclass or not
    """

    if not os.path.exists(norm_path):

        trnfsm = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor()])

        train_dataset = ImageDataset(train_dir, mc_train_data, trnfsm, multiclass=multiclass)
        train_loader = DataLoader(train_dataset, batch_size=25, num_workers=0,
                            shuffle=True)

        train_mean, train_std = get_transform_normalize_values(trnfsm, train_loader)

        train_normalize = transforms.Normalize(mean=train_mean, std=train_std)

        with open(norm_path, 'wb') as f:
            pickle.dump(train_normalize, f)

    else:
        with open(norm_path, 'rb') as f:
            train_normalize = pickle.load(f)
        
    # Set transformations
    return {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomCrop((int(input_size[0] * 0.5), 
                                int(input_size[1] * 0.5))),
            transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            train_normalize

        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            train_normalize
        ])
    }
