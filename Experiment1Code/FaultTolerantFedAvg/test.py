import torch
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
from utils import check_accuracy, save_predictions_as_imgs, save_evaluation_results
import glob

#Load Test Files

test_filenames = torch.load("test_filenames.pt")
VAL_IMG_DIR = os.path.join('Medetec_foot_ulcer_224', 'test', 'images') 
VAL_MASK_DIR =  os.path.join('Medetec_foot_ulcer_224', 'test', 'labels')
IMAGE_HEIGHT = 224  # 1280 originally
IMAGE_WIDTH = 224 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Modify the CarvanaDataset to accept predefined filenames
class CarvanaTestDataset(CarvanaDataset):
    def __init__(self, filenames, image_dir, mask_dir, transform=None):
        super().__init__(image_dir, mask_dir, transform)
        self.images = filenames

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

# Create the test dataset and loader
test_ds = CarvanaTestDataset(test_filenames, VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transforms)

test_loader = DataLoader(
    test_ds,
    batch_size=1,  # or whatever batch size you want
    shuffle=False,
    num_workers=0,  # or as required
    pin_memory=True
)

print("Number of samples in test_loader:", len(test_loader.dataset))


# Test model

model = UNET(in_channels=3, out_channels=1)
model = model.to(DEVICE)

list_of_files = [fname for fname in glob.glob("./model_round_*")]
latest_round_file = max(list_of_files, key=os.path.getctime)
print("Loading pre-trained model from: ", latest_round_file)
state_dict = torch.load(latest_round_file)
model.load_state_dict(state_dict)

def test():
    
    loss, num_examples, results = check_accuracy(test_loader, model,  device=DEVICE)
    
    # Print the results for visibility (optional)
    print(f"Dice Score: {results['dice_score']:.4f}, Loss: {results['loss']:.4f}, Mean IoU: {results['miou']:.4f}")
    
    # Save the results to CSV
    round_number = latest_round_file.split("_")[-1].split(".")[0]  # Extract the round number from filename
    save_evaluation_results(results, filename="test_results.csv", client_id=0, round_number=int(round_number))
    save_predictions_as_imgs(test_loader, model, folder="test_saved_images/", device=DEVICE)

if __name__ == "__main__":
    test()
