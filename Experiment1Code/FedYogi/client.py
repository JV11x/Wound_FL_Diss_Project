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
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    save_results_to_csv,
    save_evaluation_results,
)
import torchvision.transforms.functional as TF

CLIENT_ID = int(os.environ.get("CLIENT_ID", 0))  # Get client ID from environment variable; default to 0
TOTAL_CLIENTS = int(os.environ.get("TOTAL_CLIENTS", 3))  # Total number of clients; default to 1


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
        epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        if epoch == 0:
            aggregation_round += 1

    # Save training results to CSV
    save_results_to_csv(CLIENT_ID, epoch_losses, aggregation_round, epoch_avg_loss, filename="train_results_round_.csv")

    return epoch_losses, {"loss": epoch_avg_loss}

# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
model = UNET(in_channels=3, out_channels=1).to(DEVICE)

train_loader, val_loader, test_loader  = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
        client_id=CLIENT_ID,
        total_clients=TOTAL_CLIENTS
)

test_filenames = [test_loader.dataset.dataset.images[i] for i in test_loader.dataset.indices]
torch.save(test_filenames, "test_filenames.pt")

aggregation_round = 0

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, val_loader):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)


    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        self.set_parameters(parameters)
        train(self.model, self.train_loader, self.val_loader, epochs=NUM_EPOCHS)
        return self.get_parameters(config={}), len(train_loader.dataset), {}

#Change evaluate so that a 'test' function drives      
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        print(f"[Client {self.cid}] evaluate, config: {config}")        
        loss, num_examples, results = check_accuracy(val_loader, model, device=DEVICE)
            # print some examples to a folder
        save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)
        save_evaluation_results(results, filename="eval_results_round.csv", client_id=CLIENT_ID,round_number=aggregation_round)
        return loss, num_examples, results   
    

def client_fn(cid) -> FlowerClient:
    return FlowerClient(cid, model, train_loader, val_loader)


# Start Flower client
def main():
    fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=client_fn(cid=CLIENT_ID))
if __name__ == "__main__":
    main()