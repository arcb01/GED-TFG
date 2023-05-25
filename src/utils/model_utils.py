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
            return self.imgs_list[:, 1].astype('int32')

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.imgs_list[idx][0])
    
        labels = self.imgs_list[idx][1::]

        if self.multiclass:
            label = int(np.where(labels == 1)[0][0]) # idx
        else:
            label = float(labels[0])

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


def get_transforms(norm_path, input_size, train_dir, mc_train_data, multiclass, typ=None):
    """
    Get the data transforms
    :param norm_path: path to the normalization values
    :param input_size: resize size for the images
    :param train_dir: path to the training images folder
    :param mc_train_data: training data
    :param multiclass: whether the dataset is multiclass or not
    :param typ: type of data augmentation
    """

    if not os.path.exists(norm_path):

        trnfsm = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor()])

        train_dataset = ImageDataset(train_dir, mc_train_data, trnfsm, multiclass=multiclass)
        train_loader = DataLoader(train_dataset, batch_size=25, num_workers=0,
                            shuffle=True)

        train_mean, train_std = get_transform_normalize_values(train_loader)

        train_normalize = transforms.Normalize(mean=train_mean, std=train_std)

        with open(norm_path, 'wb') as f:
            pickle.dump(train_normalize, f)

    else:
        with open(norm_path, 'rb') as f:
            train_normalize = pickle.load(f)
        
    # Set transformations
    if typ == "FRM":
        return {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomCrop((int(input_size[0] * 0.5), 
                                int(input_size[1] * 0.5))),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAutocontrast(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            train_normalize

        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            train_normalize
        ])
    }
    elif typ == "BLR":
        return {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                transforms.GaussianBlur(13, sigma=(2.0, 6.0)),
                transforms.RandomAutocontrast(),
                transforms.ToTensor(),
                train_normalize
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                train_normalize
            ])
        }
    elif typ == "MC":
        return {
            'train': transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.RandomCrop((int(input_size[0] * 0.5), 
                                        int(input_size[1] * 0.5))),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(90),
                    transforms.RandomAutocontrast(),
                    transforms.GaussianBlur(13, sigma=(2.0, 6.0)),
                    transforms.ToTensor(),
                    train_normalize
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                train_normalize
            ])
        }


def calculate_ce_weights(data):
    """
    Calculate the class weights for each class in the dataset. 
    These weights are used to balance the loss function.
    :param data: training data
    """
    
    class_samples = []
    n_classes = len(data[:, 1::][0])
    
    for c in range(1, n_classes + 1):
        n_samples_c = data[np.where(data[:, c] == 1)].shape[0]
        class_samples.append(n_samples_c)

    total_train_samples = sum(class_samples)
    class_weights = [total_train_samples / (len(class_samples) * samples) for samples in class_samples]
    class_weights = torch.FloatTensor(class_weights)
    
    return class_weights
