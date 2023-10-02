import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import os
import pandas as pd

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    os.chmod(filename, 0o644)

def load_checkpoint(checkpoint_file, model):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location='cpu') # add map_location argument
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
    client_id=0,
    total_clients=6,
    initial_samples=300,
    shift_size=5,
    round_num=0,
    subsequent_samples=5,
):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    dataset_length_train = len(train_ds)
    if round_num == 0:
        per_client_length_train = min(subsequent_samples, dataset_length_train // total_clients)
    if round_num == 1:
        per_client_length_train = min(initial_samples, dataset_length_train // total_clients)
    else:
        per_client_length_train = min(subsequent_samples, dataset_length_train // total_clients)

    shift = (shift_size * round_num) % dataset_length_train
    start_idx_train = (client_id * per_client_length_train + shift) % dataset_length_train
    end_idx_train = (start_idx_train + per_client_length_train) % dataset_length_train

    if start_idx_train < end_idx_train:
        indices_train = list(range(start_idx_train, end_idx_train))
    else:
        indices_train = list(range(start_idx_train, dataset_length_train)) + list(range(0, end_idx_train))

    train_ds_subset = torch.utils.data.Subset(train_ds, list(range(start_idx_train, end_idx_train)))

    train_loader = DataLoader(
        train_ds_subset,  # Use the subset for training data
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    dataset_length_val = len(val_ds)
    per_client_length_val = dataset_length_val // total_clients
    start_idx_val = (client_id - 1) * per_client_length_val
    end_idx_val = start_idx_val + per_client_length_val

    val_ds_subset = torch.utils.data.Subset(val_ds, list(range(start_idx_val, end_idx_val)))

    # After:
    val_size = int(0.5 * dataset_length_val)
    test_size = dataset_length_val - val_size

    val_ds, test_ds = torch.utils.data.random_split(val_ds, [val_size, test_size])

    val_loader = DataLoader(
        val_ds_subset,  # Use the subset for validation data
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader 




def check_accuracy(loader, model, device="cuda" if torch.cuda.is_available() else "cpu"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()

    total_loss = 0
    total_intersection = 0
    total_union = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = model(x)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()
            
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

            # Compute Intersection and Union for mIoU
            intersection = (preds * y).sum()
            union = (preds + y).sum() - intersection

            total_intersection += intersection
            total_union += union

    acc = num_correct / num_pixels
    dice_score = dice_score / len(loader)
    miou = total_intersection / total_union

    model.train()
    loss = 1 - acc 

    num_examples = len(loader) 

    results = {
        "dice_score": dice_score.item(), 
        "loss": loss.item(),
        "miou": miou.item()   # convert Tensor to Python scalar
    }

    return loss.item(), num_examples, results  # convert Tensor to Python scalar



def save_predictions_as_imgs(loader, model, folder="saved_images", device="cpu"):
    model.eval()

    # Create the directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        # removed the trailing slash from folder string
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/truth_{idx}.png")


    model.train()


def save_results_to_csv(client_ids, epoch_losses, aggregation_round, avg_loss, filename="train_results.csv"):
    # Prepare data to be saved
    data = {
        "aggregation_round": aggregation_round,
        "epoch_num": list(range(1, len(epoch_losses) + 1)),  # added epoch number
        "client_id": client_ids,
        "epoch_loss": epoch_losses,
        "avg_loss": [avg_loss for _ in range(len(epoch_losses))]
    }
    df = pd.DataFrame(data)

    # If file doesn't exist, write with header, otherwise append without header
    if not os.path.isfile(filename):
        df.to_csv(filename, mode='w', index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)

def save_evaluation_results(results, filename="eval_results_round.csv", client_id=0, round_number=0):
    # If file doesn't exist, write with header, otherwise append without header
    if not os.path.isfile(filename):
        df = pd.DataFrame({"Evaluation Round": [round_number],
                            "Client ID": [client_id], 
                            "Mean IoU": [results["miou"]],
                            "Dice Score": [results["dice_score"]],
                            "Loss": [results["loss"]]
                           })  # Added client_id column
        df.to_csv(filename, mode='w', index=False)
    else:
        df = pd.DataFrame({"Evaluation Round": [round_number],
                            "Client ID": [client_id], 
                            "Mean IoU": [results["miou"]],
                            "Dice Score": [results["dice_score"]],
                            "Loss": [results["loss"]]
                           })   # Added client_id column
        df.to_csv(filename, mode='a', header=False, index=False)