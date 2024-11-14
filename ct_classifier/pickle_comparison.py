import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from train import create_dataloader, load_model
import yaml

# Load the config and dataset
config = 'all_model_states/a-resnet18_d-checked_b-128_n-50_padded_images_sharpness_medium/config_a-resnet18_d-checked_b-128_n-50_padded_images_sharpness_medium.yaml'
cfg = yaml.safe_load(open(config, 'r'))
dataset = create_dataloader(cfg, split='val').dataset  # Access the validation dataset
dl_val = DataLoader(dataset, batch_size=32, shuffle=False)  # DataLoader without shuffling
model, _ = load_model(cfg, load_latest_version=True)  # Load the model
model_device = next(model.parameters()).device

# Initialize an empty set to store unique class indices
unique_class_indices = set()

# Iterate over the validation dataset
for batch_idx, (inputs, labels) in enumerate(dl_val):
    # Move inputs and labels to the correct device
    inputs, labels = inputs.to(model_device), labels.to(model_device)

    # Get predictions
    predictions = model(inputs)
    probabilities = F.softmax(predictions, dim=1)
    _, predicted_class_indices = probabilities.max(dim=1)

    # Collect unique class indices
    unique_class_indices.update(predicted_class_indices.cpu().numpy())

# Convert the unique class indices to a sorted list (optional)
unique_class_indices = sorted(list(unique_class_indices))

# Print the unique class indices to verify
print("Unique class indices found by the model:", unique_class_indices)

# You can now create a new class mapping dictionary if you know the labels for these indices
# Assuming you have a list of class names like `class_names`
# Example (you should replace it with your actual labels):
# class_names = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', '...']  # Add all class names
class_names = [
    'zebra_finch', 'zebra_dove', 'yellow_bellied_tits', 'siberian_rubythroat', 
    'swinhoes_whiteeye', 'red_whiskered_bulbul', 'red_billed_starling', 
    'oriental_skylark', 'oriental_magpie_robin', 'marsh_tit', 'javan_pied_starling', 
    'hwamei', 'japanese_grosbeak', 'hill_mynas', 'great_myna', 'fischers_lovebird', 
    'crested_lark', 'common_redpoll', 'common_myna', 'chestnut_munia', 'cf_white_eye', 
    'bw_myna', 'bnoriole', 'bluethroat', 'bm_leafbird', 'Hoopoe', 'Grey_Parrot', 
    'Eurasian_siskin', 'Eurasian_jay'
]

# Create the new class mapping dictionary
new_class_mapping = {idx: class_names[idx] for idx in unique_class_indices}

# Save the new class mapping to a pickle file
new_class_mapping_file = 'new_class_mapping.pkl'
with open(new_class_mapping_file, 'wb') as f:
    pickle.dump(new_class_mapping, f)

print(f"New class mapping saved to {new_class_mapping_file}")
