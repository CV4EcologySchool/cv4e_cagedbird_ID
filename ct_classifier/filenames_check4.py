import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
import pickle
import csv
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
from train import create_dataloader, load_model
from util import *

# Parameters
config = '/home/home01/bssbf/cv4e_cagedbird_ID/all_model_states/ep75_56sp_anone_lr1e-3_snone_orig/config_ep75_56sp_anone_lr1e-3_snone_orig.yaml'

# Load config
print(f'Using config "{config}"')
cfg = yaml.safe_load(open(config, 'r'))
init_seed(cfg.get('seed', None))

# Setup dataloader
dl_val = create_dataloader(cfg, split='val')

# Load model
model, start_epoch = load_model(cfg, load_latest_version=True)
print(start_epoch)

# Lists to store data
inputs_list = []
labels_list = []
pred_list = []
confidence_score_list = []
mismatch_list = []

# Load the class mapping
class_mapping_file = '/home/home01/bssbf/cv4e_cagedbird_ID/ct_classifier/class_mapping_56.pickle'
with open(class_mapping_file, 'rb') as f:
    class_mapping = pickle.load(f)
print(class_mapping)

# Output directories for matches and mismatches
output_dir_match = 'predicted_images3/match'
output_dir_mismatch = 'predicted_images3/mismatch'
os.makedirs(output_dir_match, exist_ok=True)
os.makedirs(output_dir_mismatch, exist_ok=True)

# Iterate over validation data
for batch_idx, (inputs, labels) in enumerate(dl_val):
    predictions = model(inputs)
    probabilities = F.softmax(predictions, dim=1)
    max_pred, argmax_pred = probabilities.max(dim=1)
    
    for idx, (pred, true, score) in enumerate(zip(argmax_pred, labels, max_pred)):
        # Determine if the prediction is a match or mismatch
        is_mismatch = pred != true
        confidence_score_list.append(score.item())
        mismatch_list.append('Mismatch' if is_mismatch else 'Match')
        
        # Get the true and predicted label names
        true_label_name = class_mapping.get(true.item(), true.item())
        pred_label_name = class_mapping.get(pred.item(), pred.item())
        
        # Prepare the save path based on match/mismatch
        save_dir = output_dir_mismatch if is_mismatch else output_dir_match
        filename = f"true_{true_label_name}_pred_{pred_label_name}_conf_{score.item():.2f}_batch{batch_idx}_img{idx}.png"
        save_path = os.path.join(save_dir, filename)
        
        # Save the image
        img = inputs[idx].permute(1, 2, 0).cpu().numpy()
        plt.imshow(img)
        plt.title(f"Predicted: {pred_label_name}, True: {true_label_name}")
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()
        
        # Extend lists
        pred_list.append(pred)
        inputs_list.append(inputs[idx])
        labels_list.append(labels[idx])

# Map predicted and true labels
predicted_labels = [class_mapping.get(pred.item(), pred.item()) for pred in pred_list]
true_labels = [class_mapping.get(label.item(), label.item()) for label in labels_list]

# Save predictions and results to CSV with an additional 'Image Filename' column
with open('validation_predictions10.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['True Label', 'Predicted Label', 'Confidence Score', 'Mismatch', 'Image Filename'])
    for idx, (true_label, pred_label, score, mismatch) in enumerate(zip(true_labels, predicted_labels, confidence_score_list, mismatch_list)):
        # Determine the saved image path
        is_mismatch = mismatch == 'Mismatch'
        save_dir = output_dir_mismatch if is_mismatch else output_dir_match
        filename = f"true_{true_label}_pred_{pred_label}_conf_{score:.2f}_batch{batch_idx}_img{idx}.png"
        save_path = os.path.join(save_dir, filename)
        
        # Write to CSV with the filename as an extra column
        writer.writerow([true_label, pred_label, score, mismatch, filename])

print("Images and CSV file saved successfully with filenames.")