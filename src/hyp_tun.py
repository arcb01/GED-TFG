from __future__ import print_function 
from __future__ import division
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import secrets, hashlib
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
from PIL import Image
import os
from utils.model_utils import *
import os
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsampler import ImbalancedDatasetSampler
import torch.utils.data
from torchvision import transforms
from sqlalchemy import create_engine
from torch.optim.lr_scheduler import StepLR
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


def objective(trial):
    """
    Objective function for Optuna to optimize.
    """

    # Initialize the model
    model_name = "convnext"
    model, _ = initialize_model(num_classes, model_name)
    #print(model)

    # Send the model to GPU
    model = model.to(device)

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    wd = trial.suggest_float("wd", 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "RMSprop"])
    EPOCHS = trial.suggest_int("epochs", 1, 10)

    # Load data
    # Path to the file with the normalization values
    norm_path = f'./outputs/norms_MC.pkl'
    input_size = (224,224)
    # Get dict with data transforms
    data_transforms = get_transforms(norm_path, input_size, train_dir, mc_train_data, multiclass=multiclass)
    # Balance data giving corresponding weights to each class
    train_dataset = ImageDataset(train_dir, mc_train_data, data_transforms["train"], 
                                 multiclass=multiclass)
    val_dataset = ImageDataset(val_dir, mc_val_data, data_transforms["val"],
                               multiclass=multiclass)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                          num_workers=0, pin_memory=True,
                          sampler=ImbalancedDatasetSampler(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=0, pin_memory=True,
                            sampler=ImbalancedDatasetSampler(val_dataset))

    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)

            if not multiclass:
                target = target.unsqueeze(1)
                target = target.float()

            loss = loss_fn(output.float(), target)

            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)

                if not multiclass:
                    target = target.unsqueeze(1)
                    target = target.float()

                loss = loss_fn(output.float(), target)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += torch.sum(pred == target.data)
                val_loss += loss.item()

        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return val_loss


#if __name__ == "__main__":
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load datas
file = f'./data/vw_MC_dataset.json'
multiclass = True if "mc" in file.lower() else False

if multiclass:
    train_dir = '/media/arnau/SSD/VizWiz/models/multiclass/train/'
    val_dir = '/media/arnau/SSD/VizWiz/models/multiclass/val/'
    loss_fn = nn.CrossEntropyLoss()
else:
    train_dir = '/media/arnau/SSD/VizWiz/data/captioning/train/'
    val_dir = '/media/arnau/SSD/VizWiz/data/captioning/val/'
    loss_fn = nn.BCEWithLogitsLoss()

with open(file, encoding='UTF-8') as m_json_file:
    data = json.load(m_json_file)
    mc_train_data = np.array(data["train"], dtype=object)
    mc_val_data = np.array(data["val"], dtype=object)
    mc_test_data = np.array(data["test"], dtype=object)

# Number of classes in the dataset
num_classes = len(mc_train_data[:, 1::][0])

# Replace the database_name with the name of your SQLite database file
engine = create_engine('sqlite:///blur_model.db', echo=True)

# Create a studysss
h = hashlib.sha256(secrets.token_bytes(16)).hexdigest()[:5]

study = optuna.create_study(direction="minimize", 
                            storage="sqlite:///fsrm_model.db",  # Specify the storage URL here.
                            study_name=f"model-{h}",
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

