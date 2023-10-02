from typing import List
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
import flwr as fl
import os
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    save_results_to_csv,
    save_evaluation_results,
)
import torchvision.transforms.functional as TF

CLIENT_ID = int(os.environ.get("CLIENT_ID", 0))  # Get client ID from environment variable; default to 0
TOTAL_CLIENTS = int(os.environ.get("TOTAL_CLIENTS", 1))  # Total number of clients; default to 1


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 2
NUM_WORKERS = 0
IMAGE_HEIGHT = 224  # 1280 originally
IMAGE_WIDTH = 224  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = os.path.join('Medetec_foot_ulcer_224', 'train', 'images') 
TRAIN_MASK_DIR = os.path.join('Medetec_foot_ulcer_224', 'train', 'labels') 
VAL_IMG_DIR = os.path.join('Medetec_foot_ulcer_224', 'test', 'images') 
VAL_MASK_DIR =  os.path.join('Medetec_foot_ulcer_224', 'test', 'labels')
train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )



def train(model, train_loader, val_loader, epochs):
    
    global aggregation_round
    model = model.to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth"), model)
        
    # Assuming val_loader is defined somewhere
    check_accuracy(val_loader, model, device=DEVICE)

    scaler = None  # changed from torch.cuda.amp.GradScaler()
    epoch_losses = []
    
    for epoch in range(epochs):
        loop = tqdm(train_loader)
        losses = []

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.float().unsqueeze(1).to(device=DEVICE)

            # forward
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            losses.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())

        epoch_losses.append(sum(losses) / len(losses))

        save_checkpoint(epoch, model, filename="my_checkpoint.pth.tar" )

    epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
    #save_results_to_csv(epoch_losses, epoch_avg_loss, filename="train_results_round_.csv")

    return epoch_losses, {"loss": epoch_avg_loss}



# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

def get_batch_loaders(
    batch_size,
    total_batches,
    train_dir=TRAIN_IMG_DIR,
    train_maskdir=TRAIN_MASK_DIR,
    val_dir=VAL_IMG_DIR,
    val_maskdir=VAL_MASK_DIR,
    train_transform=train_transform,
    val_transform=val_transforms,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
):
    for batch_id in range(total_batches):
        train_loader, val_loader, test_loader = get_loaders(
            train_dir,
            train_maskdir,
            val_dir,
            val_maskdir,
            batch_size,
            train_transform,
            val_transform,
            num_workers,
            pin_memory,
            client_id=batch_id+1,
            total_clients=total_batches  # Using total_batches to split data
        )
        yield train_loader, val_loader, train_loader
        test_filenames = [test_loader.dataset.dataset.images[i] for i in test_loader.dataset.indices]
        torch.save(test_filenames, "test_filenames.pt")

def scenario_two(total_batches):
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    
    for batch_idx, (train_loader, val_loader, _) in enumerate(get_batch_loaders(BATCH_SIZE, total_batches)):
        print(f"Training on batch {batch_idx + 1}...")
        train(model, train_loader, val_loader, NUM_EPOCHS)

if __name__ == "__main__":
    scenario_two(total_batches=3)
