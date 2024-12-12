import torch
import yaml
import pickle
import csv
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from train import create_dataloader, load_model
from util import init_seed
import os
from PIL import Image

# Parameters
config = 'all_model_states/a-resnet18_d-checked_b-128_n-50_padded_images_sharpness_medium/config_a-resnet18_d-checked_b-128_n-50_padded_images_sharpness_medium.yaml'
output_folder = 'predicted_images'  # Main folder to save the images

# Subfolders for match and mismatch images
match_folder = os.path.join(output_folder, 'match')
mismatch_folder = os.path.join(output_folder, 'mismatch')

# Create subfolders if they don't exist
os.makedirs(match_folder, exist_ok=True)
os.makedirs(mismatch_folder, exist_ok=True)

# Load config
print(f'Using config "{config}"')
cfg = yaml.safe_load(open(config, 'r'))

# Set the seed, so you can reproduce the randomness
init_seed(cfg.get('seed', None))

# Setup dataset and manually create a DataLoader with shuffle=False
dataset = create_dataloader(cfg, split='val').dataset  # Access the dataset from `create_dataloader`
dl_val = DataLoader(dataset, batch_size=32, shuffle=False)  # Create a new DataLoader with shuffle=False

# Load model
model, start_epoch = load_model(cfg, load_latest_version=True)
print(f"Model loaded from epoch {start_epoch}")

# Get model device from one of its parameters
model_device = next(model.parameters()).device

# Predict and save results
confidence_score_list = []
mismatch_list = []

# Load the class mapping dictionary
class_mapping_file = '/home/home01/bssbf/cv4e_cagedbird_ID/new_class_mapping.pkl'
with open(class_mapping_file, 'rb') as f:
    class_mapping = pickle.load(f)

# Define image transformation for saving images
transform = ToPILImage()

# Open the CSV file to write predictions (excluding filename)
with open('validation_predictions4.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Row Number', 'True Label', 'Predicted Label', 'Confidence Score', 'Mismatch'])  # Exclude filename

    # Iterate over validation data without shuffle to preserve order
    image_index = 0
    for batch_idx, (inputs, labels) in enumerate(dl_val):
        # Move inputs to the model's device
        inputs, labels = inputs.to(model_device), labels.to(model_device)
        
        predictions = model(inputs)
        probabilities = F.softmax(predictions, dim=1)
        max_pred, argmax_pred = probabilities.max(dim=1)

        for idx, (input_img, pred, true, score) in enumerate(zip(inputs, argmax_pred, labels, max_pred)):
            accuracy = 1 if pred == true else 0
            confidence_score_list.append(score.item())
            mismatch_list.append('Mismatch' if pred != true else 'Match')

            # Retrieve the predicted and true labels
            pred_label = class_mapping.get(pred.item(), 'Unknown')
            true_label = class_mapping.get(true.item(), 'Unknown')

            # Print debug information for the image and its label
            print(f"Processing image {image_index} - Pred: {pred_label}, True: {true_label}, Score: {score.item():.2f}")

            # Convert tensor image to PIL for saving (ensure it's on CPU)
            img = transform(input_img.cpu())

            # Determine folder based on match or mismatch
            folder = match_folder if pred == true else mismatch_folder

            # Generate a unique filename for each image based on the index
            img_filename = f'image_{image_index}_pred_{pred_label}_true_{true_label}_score_{score.item():.2f}.png'
            img_path = os.path.join(folder, img_filename)

            # Save the image
            img.save(img_path)

            # Write the image details (excluding filename) to the CSV
            writer.writerow([image_index, true_label, pred_label, score.item(), 'Mismatch' if pred != true else 'Match'])

            print(f"Saved image {img_path} - Pred: {pred_label}, True: {true_label}")

            image_index += 1  # Increment image index for unique filenames