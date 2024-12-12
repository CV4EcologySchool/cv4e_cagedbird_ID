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
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, precision_recall_curve, average_precision_score
from train import create_dataloader, load_model
from util import *

# Parameters
config = '/home/home01/bssbf/cv4e_cagedbird_ID/all_model_states/ep100_56sp_ahorflip0.5_lr1e-2_snone_orig/config_ep100_56sp_ahorflip0.5_lr1e-2_snone_orig.yaml'

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
filename_list = []

# Load the class mapping
class_mapping_file = '/home/home01/bssbf/cv4e_cagedbird_ID/ct_classifier/class_mapping_56.pickle'
with open(class_mapping_file, 'rb') as f:
    class_mapping = pickle.load(f)
print(class_mapping)

# Output directories for annotated and raw images
output_dir_match = 'predicted_images5/match'
output_dir_mismatch = 'predicted_images5/mismatch'
output_dir_match_raw = 'predicted_images5/match_raw'
output_dir_mismatch_raw = 'predicted_images5/mismatch_raw'
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
        is_mismatch = pred != true
        confidence_score_list.append(score.item())
        mismatch_list.append('Mismatch' if is_mismatch else 'Match')

        # Get true and predicted label names
        true_label_name = class_mapping.get(true.item(), true.item())
        pred_label_name = class_mapping.get(pred.item(), pred.item())

        # Prepare save paths
        save_dir = output_dir_mismatch if is_mismatch else output_dir_match
        filename = f"true_{true_label_name}_pred_{pred_label_name}_conf_{score.item():.2f}_batch{batch_idx}_img{idx}.png"
        save_path = os.path.join(save_dir, filename)

        # Save annotated image
        img = inputs[idx].permute(1, 2, 0).cpu().numpy()
        plt.imshow(img)
        plt.title(f"Predicted: {pred_label_name}, True: {true_label_name}")
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()

        # Save raw image
        raw_dir = output_dir_mismatch_raw if is_mismatch else output_dir_match_raw
        raw_save_path = os.path.join(raw_dir, filename)
        plt.imsave(raw_save_path, img)

        # Save filename for CSV output
        filename_list.append(filename)

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

# Save predictions and results to CSV
with open('validation_predictions13.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['True Label', 'Predicted Label', 'Confidence Score', 'Mismatch', 'Image Filename'])
    for true_label, pred_label, score, mismatch, filename in zip(true_labels, predicted_labels, confidence_score_list, mismatch_list, filename_list):
        writer.writerow([true_label, pred_label, score, mismatch, filename])

# === Classification Metrics ===
true_labels_numeric = [label.item() for label in labels_list]
predicted_labels_numeric = [pred.item() for pred in pred_list]

# Per-class metrics
precision, recall, f1, _ = precision_recall_fscore_support(true_labels_numeric, predicted_labels_numeric, average=None)
class_metrics_df = pd.DataFrame({
    "Species": list(class_mapping.values()),
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
})
class_metrics_df.to_csv("species_class_metrics2.csv", index=False)

# Overall metrics
precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(true_labels_numeric, predicted_labels_numeric, average='weighted')
print(f"Overall Precision: {precision_avg:.3f}")
print(f"Overall Recall: {recall_avg:.3f}")
print(f"Overall F1 Score: {f1_avg:.3f}")

# Generate classification report
report = classification_report(true_labels_numeric, predicted_labels_numeric, target_names=class_mapping.values())
print(report)

# === Confusion Matrix ===
conf_matrix = confusion_matrix(true_labels_numeric, predicted_labels_numeric, normalize='true')
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt='.2f', xticklabels=class_mapping.values(), yticklabels=class_mapping.values())
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("confusion_matrix_normalized2.png")
plt.show()

# === Histograms for Problematic Species ===
low_f1_species = class_metrics_df[class_metrics_df['F1 Score'] < 0.7]['Species'].tolist()
for species in low_f1_species:
    species_df = pd.DataFrame({
        "True Label": true_labels,
        "Predicted Label": predicted_labels,
        "Confidence Score": confidence_score_list,
        "Mismatch": mismatch_list
    })
    species_df = species_df[species_df["True Label"] == species]

    plt.figure(figsize=(10, 6))
    sns.histplot(species_df['Confidence Score'], bins=20, kde=True)
    plt.title(f"Confidence Scores for {species}")
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"histogram_{species}.png")
    plt.show()

# Summary of matches and mismatches
print(f"Total Matches: {match_count}")
print(f"Total Mismatches: {mismatch_count}")

# PR curve generation
# Collect ground truth labels and prediction scores

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# Collect ground truth labels and prediction scores
all_true_labels = []
all_pred_scores = []

for batch_idx, (inputs, labels) in enumerate(dl_val):
    with torch.no_grad():
        predictions = model(inputs)
        probabilities = F.softmax(predictions, dim=1)

    all_true_labels.extend(labels.cpu().numpy())
    all_pred_scores.extend(probabilities.cpu().numpy())

all_true_labels = np.array(all_true_labels)
all_pred_scores = np.array(all_pred_scores)

# Binarize true labels for multi-class PR curve
num_classes = all_pred_scores.shape[1]
binary_true_labels = label_binarize(all_true_labels, classes=range(num_classes))

# Compute micro-averaged Precision-Recall curve
precision, recall, _ = precision_recall_curve(binary_true_labels.ravel(), all_pred_scores.ravel())
mean_ap = average_precision_score(binary_true_labels, all_pred_scores, average='micro')

# Plot the Precision-Recall curve
plt.figure(figsize=(10, 8))
plt.plot(recall, precision, label=f"Micro-averaged PR Curve (AP = {mean_ap:.2f})")
plt.title("Precision-Recall Curve (Micro-averaged)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.grid()
plt.savefig("precision_recall_curve_micro2.png")
plt.show()

print(f"Micro-averaged Mean Average Precision (mAP): {mean_ap:.2f}")
