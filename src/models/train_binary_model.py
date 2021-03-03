import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

def train_binary(train_data, model, criterion, optimizer, batch_size, device, scheduler=None, generate_batch=None):
    """Train a Pytorch binary classification model

    Parameters
    ----------
    train_data : torch.utils.data.Dataset
        Pytorch dataset
    model: torch.nn.Module
        Pytorch Model
    criterion: function
        Loss function
    optimizer: torch.optim
        Optimizer
    bacth_size : int
        Number of observations per batch
    device : str
        Name of the device used for the model
    scheduler : torch.optim.lr_scheduler
        Pytorch Scheduler used for updating learning rate
    collate_fn : function
        Function defining required pre-processing steps

    Returns
    -------
    Float
        Loss score
    Float:
        Accuracy Score
    """
    
    # Set model to training mode
    model.train()
    train_loss = 0
    train_acc = 0
    
    # Create data loader
    data = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)
    
    # Iterate through data by batch of observations
    for feature, target_class in data:

        # Reset gradients
        optimizer.zero_grad()
        
        # Load data to specified device
        feature, target_class = feature.to(device), target_class.to(device).to(torch.float32)
        
        # Make predictions
        output = model(feature)
        
        # Calculate loss for given batch
        loss = criterion(output, target_class.unsqueeze(1))
        
        # Calculate global loss
        train_loss += loss.item()
        
        # Calculate gradients
        loss.backward()

        # Update Weights
        optimizer.step()
        
        # Calculate global accuracy
        train_acc += (output.argmax(1) == target_class).sum().item()

    # Adjust the learning rate
    if scheduler:
        scheduler.step()

    return train_loss / len(train_data), train_acc / len(train_data)
