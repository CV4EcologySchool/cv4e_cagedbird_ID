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

# Output directories for annotated and raw images
output_dir_match = 'predicted_images3/match'
output_dir_mismatch = 'predicted_images3/mismatch'
output_dir_match_raw = 'predicted_images3/match_raw'
output_dir_mismatch_raw = 'predicted_images3/mismatch_raw'
os.makedirs(output_dir_match, exist_ok=True)
os.makedirs(output_dir_mismatch, exist_ok=True)
os.makedirs(output_dir_match_raw, exist_ok=True)
os.makedirs(output_dir_mismatch_raw, exist_ok=True)

# Counters for matches and mismatches
match_count = 0
mismatch_count = 0

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
        
        # Prepare the save path for annotated images
        save_dir = output_dir_mismatch if is_mismatch else output_dir_match
        filename = f"true_{true_label_name}_pred_{pred_label_name}_conf_{score.item():.2f}_batch{batch_idx}_img{idx}.png"
        save_path = os.path.join(save_dir, filename)
        
        # Save the annotated image
        img = inputs[idx].permute(1, 2, 0).cpu().numpy()
        plt.imshow(img)
        plt.title(f"Predicted: {pred_label_name}, True: {true_label_name}")
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()
        
        # Save the raw image with the same filename
        raw_dir = output_dir_mismatch_raw if is_mismatch else output_dir_match_raw
        raw_save_path = os.path.join(raw_dir, filename)  # Use the same filename
        plt.imsave(raw_save_path, img)
        
        # Update counters
        if is_mismatch:
            mismatch_count += 1
        else:
            match_count += 1
        
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
        writer.writerow([true_label, pred_label, score, mismatch, filename])

# === Confusion Matrix Modification ===

# Create a DataFrame for mismatches
mismatch_df = pd.DataFrame({
    'True Label': true_labels,
    'Predicted Label': predicted_labels,
    'Mismatch': mismatch_list
})
mismatch_df = mismatch_df[mismatch_df['Mismatch'] == 'Mismatch']

# Count occurrences of each mismatch pair
confusion_pairs = mismatch_df.groupby(['True Label', 'Predicted Label']).size().reset_index(name='Count')

# Filter for mismatches with Count > 1
filtered_confusion_pairs = confusion_pairs[confusion_pairs['Count'] > 1]

# Create a filtered confusion matrix
filtered_confusion_matrix = filtered_confusion_pairs.pivot(index='True Label', columns='Predicted Label', values='Count').fillna(0)

# Plot the filtered heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(filtered_confusion_matrix, annot=True, cmap="YlOrRd", fmt='g', cbar=True)
plt.title("Heatmap of Species Confusions (Mismatches > 1)")
plt.xlabel("Predicted Species")
plt.ylabel("True Species")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save the heatmap to a PNG image
plt.savefig("species_confusion_heatmap_filtered.png")
plt.show()

# Save the filtered confusion pairs to a CSV file
filtered_confusion_pairs.to_csv("filtered_species_confusion_summary.csv", index=False)

# Print the total number of matches and mismatches
print(f"Total Matches: {match_count}")
print(f"Total Mismatches: {mismatch_count}")
print("Images, CSV files, and heatmap saved successfully.")