import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import yaml
import pickle
from train import create_dataloader, load_model

# Load the configuration and dataset
config = 'all_model_states/a-resnet18_d-checked_b-128_n-50_padded_images_sharpness_medium/config_a-resnet18_d-checked_b-128_n-50_padded_images_sharpness_medium.yaml'
cfg = yaml.safe_load(open(config, 'r'))
dataset = create_dataloader(cfg, split='val').dataset  # Access the validation dataset
subset_indices = torch.randperm(len(dataset))[:12]  # Take a random subset of 12 images
subset = Subset(dataset, subset_indices)  # Create subset dataset
dl_val = DataLoader(subset, batch_size=12, shuffle=False)  # DataLoader without shuffling

# Load the model
model, _ = load_model(cfg, load_latest_version=True)
model.eval()
model_device = next(model.parameters()).device
model.to(model_device)

# Load class mapping
class_mapping_file = '/home/home01/bssbf/cv4e_cagedbird_ID/new_class_mapping.pkl'
with open(class_mapping_file, 'rb') as f:
    class_mapping = pickle.load(f)

# Visualize the predictions
with torch.no_grad():
    for inputs, labels in dl_val:
        inputs, labels = inputs.to(model_device), labels.to(model_device)
        
        # Get predictions and convert them to probabilities
        predictions = model(inputs)
        probabilities = F.softmax(predictions, dim=1)
        _, preds = probabilities.max(dim=1)
        
        # Plot images with true and predicted labels
        fig = plt.figure(figsize=(12, 8))
        for idx in range(len(inputs)):
            ax = fig.add_subplot(3, 4, idx + 1, xticks=[], yticks=[])
            
            # Display the image
            img = inputs[idx].cpu().permute(1, 2, 0)  # Convert from CxHxW to HxWxC
            ax.imshow(img)
            
            # Get the true and predicted labels
            true_label = class_mapping.get(labels[idx].item(), 'Unknown')
            pred_label = class_mapping.get(preds[idx].item(), 'Unknown')
            
            # Set title with true and predicted labels
            ax.set_title(f"True: {true_label}\nPred: {pred_label}", color="green" if true_label == pred_label else "red")
        
        plt.tight_layout()
        plt.savefig("val_loader_predictions.png")  # Save the figure
        plt.show()
        break  # Only visualize one batch of 12 images
