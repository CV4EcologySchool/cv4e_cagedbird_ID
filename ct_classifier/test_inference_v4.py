import os
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import average_precision_score
import yaml
import pickle
from train_save_epoch import create_dataloader, load_model

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
model.eval()  # Set the model to evaluation mode


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

# Check folder structure and verify that each species has its own folder
test_set_dir = '/home/home01/bssbf/cv4e_cagedbird_ID/test/'  # Adjust to the path of your test set

for species in test_species:
    species_folder = os.path.join(test_set_dir, species)
    if not os.path.exists(species_folder):
        print(f"Warning: Folder for species '{species}' not found in the test set.")
    else:
        print(f"Folder for species '{species}' found.")

# Lists to store data for checking alignment
inputs_list = []
labels_list = []
pred_list = []
filename_list = []

# Evaluate model
def evaluate_model(model, dl_test, test_class_indices):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    all_filenames = []

    # Iterate over the test dataset
    for batch_idx, (inputs, labels, filenames) in enumerate(dl_test):
        inputs = inputs.to(device)  # Send inputs to the same device as the model
        labels = labels.to(device)  # Send labels to the same device as the model

        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)

        all_labels.append(labels)
        all_preds.append(preds)
        all_probs.append(probs)
        all_filenames.extend(filenames)

        print(f"Processed batch {batch_idx + 1}, Batch size: {labels.size(0)}")

    # Flatten the lists
    all_labels = torch.cat(all_labels).cpu().numpy()
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_probs = torch.cat(all_probs).detach().cpu().numpy()

    # Filter out the test classes
    filtered_labels = all_labels[np.isin(all_labels, test_class_indices)]
    filtered_preds = all_preds[np.isin(all_preds, test_class_indices)]
    filtered_probs = all_probs[np.isin(all_labels, test_class_indices)]

    # Ensure predictions are within valid class indices
    filtered_preds = np.clip(filtered_preds, 0, len(test_class_indices) - 1)

    # Calculate AUPRC for each class
    one_hot_labels = np.eye(len(test_class_indices))[filtered_labels]
    one_hot_preds = np.eye(len(test_class_indices))[filtered_preds]
    auprc = average_precision_score(one_hot_labels, one_hot_preds, average=None, zero_division=1)

    print(f"AUPRC for each class: {auprc}")

    # Save results to a CSV for later analysis
    with open('prediction_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'True Label', 'Predicted Label'])
        for filename, true_label, pred_label in zip(all_filenames, filtered_labels, filtered_preds):
            writer.writerow([filename, test_class_mapping[true_label], test_class_mapping[pred_label]])

    return auprc

# Call the evaluate_model function
auprc = evaluate_model(model, dl_test, test_class_indices)
print(f"Overall AUPRC: {auprc}")
