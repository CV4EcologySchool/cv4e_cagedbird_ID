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

# Parameters, config has been edited to include a test root
config = '/home/home01/bssbf/cv4e_cagedbird_ID/all_model_states/ep100_56sp_ahorflip0.5_lr1e-2_snone_orig/config_ep100_56sp_ahorflip0.5_lr1e-2_snone_orig.yaml'

# Load config
print(f'Using config "{config}"')
cfg = yaml.safe_load(open(config, 'r'))
init_seed(cfg.get('seed', None))

# Setup dataloader
dl_test = create_dataloader(cfg, split='test')

# Load model
model, start_epoch = load_model(cfg, load_latest_version=True)
print(f"Resuming from epoch {start_epoch}")

validate(cfg, dl_test, model)

# Load the class mapping
class_mapping_file = '/home/home01/bssbf/cv4e_cagedbird_ID/ct_classifier/class_mapping_56.pickle'
with open(class_mapping_file, 'rb') as f:
    class_mapping = pickle.load(f)
print("Class mapping:", class_mapping)

# Define test species and restrict to their indices
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

# Create test class mapping and indices
test_class_mapping = {idx: species for idx, species in class_mapping.items() if species in test_species}
test_class_indices = list(test_class_mapping.keys())

print(f"Test class mapping: {test_class_mapping}")
print(f"Test class indices: {test_class_indices}")

# Lists to store data
inputs_list = []
labels_list = []
pred_list = []
confidence_score_list = []
filename_list = []

# Iterate over test data and check label-image alignment
for batch_idx, (inputs, labels) in enumerate(dl_test):
    if batch_idx == 0:  # Check the first batch only for debugging
        print(f"Inputs shape: {inputs.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Visualize first 12 images with labels
        fig = plt.figure(figsize=(12, 8))
        for idx in range(12):
            print(f"True label (idx {idx}): {test_class_mapping.get(labels[idx].item())}")
            ax = fig.add_subplot(3, 4, idx + 1, xticks=[], yticks=[])
            ax.imshow(inputs[idx].permute(1, 2, 0))  # Display the image
            ax.set_title(f"True: {test_class_mapping.get(labels[idx].item())}")
        plt.tight_layout()
        plt.show()
        plt.savefig("sample_images.png")

        break  # Stop after first batch

# Process test data and get predictions
for batch_idx, (inputs, labels) in enumerate(dl_test):
    with torch.no_grad():
        predictions = model(inputs)
        probabilities = F.softmax(predictions, dim=1)
        max_pred, argmax_pred = probabilities.max(dim=1)

    for idx, (pred, true, score) in enumerate(zip(argmax_pred, labels, max_pred)):
        # Append predictions and labels
        pred_list.append(pred.item())
        labels_list.append(true.item())
        confidence_score_list.append(score.item())

# Map predicted and true labels based on the test class mapping
predicted_labels = [test_class_mapping.get(pred, pred) for pred in pred_list]
true_labels = [test_class_mapping.get(label, label) for label in labels_list]

# Metrics Calculation
precision, recall, f1, _ = precision_recall_fscore_support(
    labels_list, pred_list, labels=test_class_indices, average=None, zero_division=0
)

# Create metrics DataFrame
class_metrics_df = pd.DataFrame({
    "Species": [test_class_mapping[idx] for idx in test_class_indices],
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
})
class_metrics_df.to_csv("species_class_metrics_test.csv", index=False)

# Overall metrics
precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
    labels_list, pred_list, average='weighted', zero_division=0
)
print(f"Overall Precision: {precision_avg:.3f}")
print(f"Overall Recall: {recall_avg:.3f}")
print(f"Overall F1 Score: {f1_avg:.3f}")

# Save predictions and results to CSV
with open('preds_test.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['True Label', 'Predicted Label', 'Confidence Score'])
    for true_label, pred_label, score in zip(true_labels, predicted_labels, confidence_score_list):
        writer.writerow([true_label, pred_label, score])
