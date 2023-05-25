from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import json
from torchsampler import ImbalancedDatasetSampler
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import random
import copy
import io
import cv2
from utils.model_utils import *
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


def train_model(model, dataloaders, loss_fn, optimizer, num_epochs, save_path):
    """
    Train the model
    :param model: model to train
    :param dataloaders: dict with train and validation dataloaders
    :param loss_fn: loss function
    :param optimizer: optimizer
    :param num_epochs: number of epochs
    :param save_path: path to save the model
    :return: model, acc_history, losses 
    """
    
    since = time.time()

    acc_history = {"train": [], "val": []}
    losses = {"train": [], "val": []}

    # we will keep a copy of the best weights so far according to validation accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    output = model(inputs)

                    if not multiclass:
                        output = output.float()
                        labels = labels.unsqueeze(1).float()

                    loss = loss_fn(output, labels)
                    losses[phase].append(loss.cpu().detach().numpy())
                    # Get the index of the max log-probability.
                    preds = torch.argmax(output, 1) # 

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                if not multiclass:
                    preds = torch.round(torch.sigmoid(output))
                    #labels = labels.data.unsqueeze(1)

                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            acc_history[phase].append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    
    torch.save(model.state_dict(), save_path)
    
    return model, acc_history, losses


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load model data
    file = f'./data/vw_FRM_dataset.json'
    multiclass = True if "mc" in file.lower() else False
    flaw = file.split("_")[1]

    with open(file, encoding='UTF-8') as m_json_file:
        data = json.load(m_json_file)
        mc_train_data = np.array(data["train"], dtype=object)
        mc_val_data = np.array(data["val"], dtype=object)
        mc_test_data = np.array(data["test"], dtype=object)

    if multiclass:
        train_dir = '/media/arnau/SSD/VizWiz/models/multiclass/train/'
        val_dir = '/media/arnau/SSD/VizWiz/models/multiclass/val/'
        class_weights = calculate_ce_weights(np.array(mc_train_data, dtype=object))
        class_weights = torch.FloatTensor(class_weights).cuda()
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        typ = "MC"
    else:
        train_dir = '/media/arnau/SSD/VizWiz/data/captioning/train/'
        val_dir = '/media/arnau/SSD/VizWiz/data/captioning/val/'
        loss_fn = nn.BCEWithLogitsLoss()
        typ = str(file.split("_")[1])

    # Number of classes in the dataset
    num_classes = len(mc_train_data[:, 1::][0])

    # Initialize the model
    model_name = "convnext"
    model, input_size = initialize_model(num_classes, model_name)
    print(model)

    model = model.to(device)

    # Model parameters
    hp = {"lr" : 7.803848286745978e-05,
        "batch_size" : 128,
        "num_epochs" : 8}

    batch_size = hp["batch_size"]
    num_epochs = hp["num_epochs"]
    lr = hp["lr"]
    #optimizer = optim.AdamW(model.parameters(), lr=lr)
    # rmsprop optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=3.067449309133902e-05)

    norm_path = f'./outputs/norms_{flaw}.pkl'
    input_size = (224,224)

    data_transforms = get_transforms(norm_path, input_size, train_dir, 
                                     mc_train_data, multiclass=multiclass, typ=flaw)

    # Balance data giving corresponding weights to each class
    train_dataset = ImageDataset(train_dir, mc_train_data, data_transforms["train"], multiclass=multiclass)
    val_dataset = ImageDataset(val_dir, mc_val_data, data_transforms["val"], multiclass=multiclass)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            num_workers=0, pin_memory=True)
                            #sampler=ImbalancedDatasetSampler(train_dataset))

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=0, pin_memory=True)
                            #sampler=ImbalancedDatasetSampler(val_dataset))

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    save_path = f'./outputs/best_{typ}_{model_name}_test.pth'

    if not os.path.exists(save_path):
        # Train and evaluate
        model, hist, losses = train_model(model, dataloaders_dict, loss_fn, optimizer, 
                                        num_epochs=num_epochs, save_path=save_path)
        
        # plot the losses and accuracies
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.set_title("Loss")
        ax1.plot(losses["train"], label="training loss")
        ax1.plot(losses["val"], label="validation loss")
        ax1.legend()

        ax2.set_title("Accuracy")
        ax2.plot([x.cpu().numpy() for x in hist["train"]],label="training accuracy")
        ax2.plot([x.cpu().numpy() for x in hist["val"]],label="val accuracy")
        ax2.set_ylim([0, 1])
        ax2.legend()

        plt.show()

