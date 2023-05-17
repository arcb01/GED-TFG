#!/usr/bin/env python
# coding: utf-8

# # Basline blur classifier model



from __future__ import print_function 
from __future__ import division
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import copy
import io
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)



# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# ## Dataset

class ImageDataset(Dataset):

    def __init__(self, root_dir, imgs_list, transform=None):

        self.root_dir = root_dir
        self.imgs_list = imgs_list
        self.transform = transform

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

         # VizWiz dataset case
        img_name = os.path.join(self.root_dir, self.imgs_list[idx][0])
    
        labels = self.imgs_list[idx][1::] # [x, y, z]  
        label = int(np.where(labels == 1)[0][0])
        
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)
        
        return image, int(label)
                

# Load datase

with open(f'./data/vw_mc_dataset.json', encoding='UTF-8') as m_json_file:
    data = json.load(m_json_file)
    mc_train_data = np.array(data["train"], dtype=object)
    mc_val_data = np.array(data["val"], dtype=object)
    mc_test_data = np.array(data["test"], dtype=object)


def initialize_model(num_classes, model_name):
    """
    Initialize blur model for binary classifaction
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


# Number of classes in the dataset
num_classes = 3

# Initialize the model
model_name = "convnext"
model, input_size = initialize_model(num_classes, model_name)
print(model)

# Initialize the model
model, input_size = initialize_model(num_classes, "convnext")

# Print the model we just instantiated
print(model)

def calculate_ce_weights(data):
    
    class_samples = []
    
    for nc in [1,2,3]:
        n_samples_c = data[np.where(data[:, nc] == 1)].shape[0]
        class_samples.append(n_samples_c)

    total_train_samples = sum(class_samples)
    class_weights = [total_train_samples / (len(class_samples) * samples) for samples in class_samples]
    class_weights = torch.FloatTensor(class_weights)
    
    return class_weights


# Send the model to GPU
model = model.to(device)

# Weighted Cross entropy loss 
class_weights = calculate_ce_weights(np.array(mc_train_data, dtype=object))
class_weights = torch.FloatTensor(class_weights).cuda()
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
#vloss_fn = nn.CrossEntropyLoss()


def get_transform_normalize_values(simple_transforms, loader):
    
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


train_dir = '/media/arnau/SSD/VizWiz/multiclass/train/'
val_dir = '/media/arnau/SSD/VizWiz/multiclass/val/'
norm_path = f'./outputs/norms.pkl'
input_size = (224,224)

if not os.path.exists(norm_path):

    trnfsm = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()])

    train_dataset = ImageDataset(train_dir, mc_train_data, trnfsm)
    train_loader = DataLoader(train_dataset, batch_size=10, num_workers=0,
                        shuffle=False)

    train_mean, train_std = get_transform_normalize_values(trnfsm, train_loader)

    train_normalize = transforms.Normalize(mean=train_mean, std=train_std)

    with open(norm_path, 'wb') as f:
        pickle.dump(train_normalize, f)

else:
    with open(norm_path, 'rb') as f:
        train_normalize = pickle.load(f)
    
# Set transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomCrop((int(input_size[0] * 0.75), 
                               int(input_size[1] * 0.75))),
        transforms.GaussianBlur(kernel_size=5, 
                                sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        train_normalize

    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        train_normalize
    ])
}

train_dataset = ImageDataset(train_dir, mc_train_data, data_transforms["train"])
val_dataset = ImageDataset(val_dir, mc_val_data, data_transforms["val"])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

dataloaders_dict = {"train": train_loader, "val": val_loader}

import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from sklearn.metrics import f1_score
from torchvision import datasets
from torchvision import transforms
from sqlalchemy import create_engine
from torch.optim.lr_scheduler import StepLR

# Replace the database_name with the name of your SQLite database file
engine = create_engine('sqlite:///blur_model.db', echo=True)
EPOCHS = 10

def objective(trial):
    # Generate the model.
    model, input_size = initialize_model(num_classes, "convnext")
    model = model.to(device)

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = 128

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, 
                              pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0,
                             pin_memory=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Define the learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output.float(), target.float())
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation of the model.
        model.eval()
        correct = 0
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_fn(output.float(), target.float())
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += torch.sum(pred == target.data)
                val_loss += loss.item()

        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return val_loss


study = optuna.create_study(direction="minimize", 
                            storage="sqlite:///frm_model.db",  # Specify the storage URL here.
                            study_name="mc-nuni-b",
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.HyperbandPruner())
    
study.optimize(objective, n_trials=100)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
