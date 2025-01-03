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
from train_save_epoch import create_dataloader, load_model
from dataset import CTDataset
from util import *

# Load the class mapping from the training set
class_mapping_file = '/home/home01/bssbf/cv4e_cagedbird_ID/ct_classifier/class_mapping_56.pickle'
with open(class_mapping_file, 'rb') as f:
    class_mapping_train = pickle.load(f)

# Test species list (from your test set)
test_species = [
    'af_bluebird', 'bali_myna', 'bc_hanging_parrot', 'bh_bulbul', 'bm_leafbird',
    'bt_laughingthrush', 'bw_leafbird', 'cc_laughing', 'cc_thrush', 'cg_magpie',
    'common_myna', 'crested_lark', 'crested_myna', 'ft_barbet', 'gf_leafbird',
    'gg_leafbird', 'hill_myna', 'hooded_butcherbird', 'hoopoe', 'hwamei',
    'jap_grosbeak', 'javan_sparrow', 'jb_pitta', 'jp_starling', 'lg_leafbird',
    'oh_thrush', 'om_robin', 'rb_leiothrix', 'rubythroat', 'rw_bulbul', 'sb_munia',
    'scarlet_minivet', 'se_mesia', 'sh_bulbul', 'spotted_dove', 'sum_laughingthrush',
    'swinhoes_white_eye', 'wc_laughingthrush', 'wh_munia', 'wr_munia', 'wr_shama',
    'yb_tit', 'zebra_dove', 'zebra_finch', 'Eurasian_jay', 'Eurasian_siskin'
]

# Create a new test mapping based on the training class mapping
test_mapping = {}
for idx, species in class_mapping_train.items():
    if species in test_species:
        test_mapping[idx] = species

# Save the new test mapping to a pickle file
test_mapping_file = '/home/home01/bssbf/cv4e_cagedbird_ID/ct_classifier/test_mapping.pickle'
with open(test_mapping_file, 'wb') as f:
    pickle.dump(test_mapping, f)

print("Test mapping saved:", test_mapping)


# Parameters, config has been edited to include a test root
config = '/home/home01/bssbf/cv4e_cagedbird_ID/all_model_states/ep100_56sp_ahorflip0.5_lr1e-2_snone_orig/config_ep100_56sp_ahorflip0.5_lr1e-2_snone_orig.yaml'

# Load config
print(f'Using config "{config}"')
cfg = yaml.safe_load(open(config, 'r'))
init_seed(cfg.get('seed', None))

# Setup dataloader
dl_test = create_dataloader(cfg, split='test')

# Load model, when training it saves the model as latest.pt in case the model cuts out at a certain number of epochs
model, start_epoch = load_model(cfg, load_latest_version=True)
print(start_epoch)

# Lists to store data
inputs_list = []
labels_list = []
pred_list = []
confidence_score_list = []
mismatch_list = []
filename_list = []

# Output directories for annotated and raw images
output_dir_match = 'test_images/match'
output_dir_mismatch = 'test_images/mismatch'
os.makedirs(output_dir_match, exist_ok=True)
os.makedirs(output_dir_mismatch, exist_ok=True)

# Counters for matches and mismatches
match_count = 0
mismatch_count = 0

# Iterate over validation data
for batch_idx, (inputs, labels) in enumerate(dl_test):
    predictions = model(inputs)
    probabilities = F.softmax(predictions, dim=1)
    max_pred, argmax_pred = probabilities.max(dim=1)

    for idx, (pred, true, score) in enumerate(zip(argmax_pred, labels, max_pred)):
        is_mismatch = pred != true
        confidence_score_list.append(score.item())
        mismatch_list.append('Mismatch' if is_mismatch else 'Match')

        # Get true and predicted label names
        true_label_name = class_mapping_train.get(true.item(), true.item())
        pred_label_name = class_mapping_train.get(pred.item(), pred.item())

        # Prepare filename for CSV output
        filename = f"true_{true_label_name}_pred_{pred_label_name}_conf_{score.item():.2f}_batch{batch_idx}_img{idx}.png"
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
predicted_labels = [class_mapping_train.get(pred.item(), pred.item()) for pred in pred_list]
true_labels = [class_mapping_train.get(label.item(), label.item()) for label in labels_list]

# Check the lengths of lists
print(f"Lengths of lists:")
print(f"True labels: {len(true_labels)}")
print(f"Predicted labels: {len(predicted_labels)}")
print(f"Confidence scores: {len(confidence_score_list)}")
print(f"Mismatch list: {len(mismatch_list)}")
print(f"Filename list: {len(filename_list)}")

# Ensure all lists have the same length
assert len(true_labels) == len(predicted_labels) == len(confidence_score_list) == len(mismatch_list) == len(filename_list), \
    "The lengths of the lists do not match!"

# Save predictions and results to CSV
with open('test_pred.csv', mode='w', newline='') as file:
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
    "Species": list(class_mapping_train.values()),
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
})
class_metrics_df.to_csv("species_class_metrics_test.csv", index=False)

# Overall metrics
precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(true_labels_numeric, predicted_labels_numeric, average='weighted')
print(f"Overall Precision: {precision_avg:.3f}")
print(f"Overall Recall: {recall_avg:.3f}")
print(f"Overall F1 Score: {f1_avg:.3f}")

# Generate classification report
report = classification_report(true_labels_numeric, predicted_labels_numeric, target_names=class_mapping_train.values())
print(report)

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

    # Save histograms (seaborn can still be used here)
    sns.histplot(species_df['Confidence Score'], bins=20, kde=True)
    plt.title(f"Confidence Scores for {species}")
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"histogram_test{species}.png")
    plt.close()

# Summary of matches and mismatches
print(f"Total Matches: {match_count}")
print(f"Total Mismatches: {mismatch_count}")
