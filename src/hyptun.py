#!/usr/bin/env python
# coding: utf-8

# # Basline blur classifier model



from __future__ import print_function 
from __future__ import division
import torch
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


# ResNet input size
input_size = (224,224)

# Just normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.4857, 0.4296, 0.3671], [0.2854, 0.2750, 0.2751])
        
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.4550, 0.4196, 0.3736], [0.2908, 0.2805, 0.2861]) 
    ]),
}


# ## Dataset


class BlurDataset(Dataset):

    def __init__(self, root_dir, imgs_list, transform=None):

        self.root_dir = root_dir
        self.imgs_list = imgs_list
        self.transform = transform

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Blur kaggle dataset case
        if type(self.imgs_list[idx]) == str:
            img_name = self.imgs_list[idx]
            img_tag = self.imgs_list[idx].split("/")[-1].split("_")[-1].split(".")[0]
            if img_tag == "S":
                img_label = 0
            else:
                img_label = 1
         # VizWiz dataset case
        else:
            img_name = os.path.join(self.root_dir, self.imgs_list[idx][0])
            img_label = self.imgs_list[idx][1]

        image = Image.open(img_name)

        if self.transform:
            sample = self.transform(image)

        return sample, img_label


# Load datase


with open('../data/vw_blur_dataset.json', encoding='UTF-8') as m_json_file:
    data = json.load(m_json_file)
    m_train_data = data["train"]
    m_val_data = data["val"]

def initialize_model(num_classes):
    # vgg16
    #model = models.vgg16(pretrained=True)
    model = models.convnext_tiny(pretrained=True)
    
    model.fc = nn.Linear(512, num_classes)# YOUR CODE HERE!
    
    input_size = 224
        
    return model, input_size


# Number of classes in the dataset
num_classes = 2

# Initialize the model
model, input_size = initialize_model(num_classes)

# Print the model we just instantiated
print(model)



# Send the model to GPU
model = model.to(device)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()



train_dataset = BlurDataset('/media/arnau/PEN/TFG/train/', m_train_data, data_transforms["train"])
val_dataset = BlurDataset('/media/arnau/PEN/TFG/val/', m_val_data, data_transforms["val"])



import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from sqlalchemy import create_engine
from torch.optim.lr_scheduler import StepLR

# Replace the database_name with the name of your SQLite database file
engine = create_engine('sqlite:///blur_model.db', echo=True)
EPOCHS = 10

def objective(trial):
    # Generate the model.
    model = models.convnext_tiny(pretrained=True).to(device)

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = 128
    weight_decay = trial.suggest_float('weight_decay', 1e-9, 1e-5, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Define the learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    model.add_module("dropout", nn.Dropout(dropout_rate))

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
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
                loss = criterion(output, target)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                val_loss += loss.item()

        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return val_loss


study = optuna.create_study(direction="minimize", 
                            storage="sqlite:///blur_model.db",  # Specify the storage URL here.
                            study_name="blur-overnight2"
                            ) 
    
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
