import os
import numpy as np
import comet_ml
import torch
import yaml
import pickle
import csv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from train import load_model  # Assuming this function loads your trained model
from util import *  # Assuming this is your custom utility file

# Parameters
config = '/home/home01/bssbf/cv4e_cagedbird_ID/all_model_states/ep75_56sp_anone_lr1e-3_snone_orig/config_ep75_56sp_anone_lr1e-3_snone_orig.yaml'
image_folder = '/home/home01/bssbf/cv4e_cagedbird_ID/unannotated_test'  # Path to the folder containing unannotated images

# Load config
print(f'Using config "{config}"')
cfg = yaml.safe_load(open(config, 'r'))
init_seed(cfg.get('seed', None))

# Load model
model, _ = load_model(cfg, load_latest_version=True)
print(f'Model loaded: {model}')

# Lists to store data
inputs_list = []
pred_list = []
confidence_score_list = []

# Load the class mapping
class_mapping_file = '/home/home01/bssbf/cv4e_cagedbird_ID/ct_classifier/class_mapping_56.pickle'
with open(class_mapping_file, 'rb') as f:
    class_mapping = pickle.load(f)
print(f'Class mapping loaded: {class_mapping}')

# Output directories for predicted images
output_dir = 'predicted_images'
os.makedirs(output_dir, exist_ok=True)

# Define the image transformations manually
def transform_image(image, target_size=(224, 224)):
    # Resize the image using PIL
    image = image.resize(target_size)
    
    # Convert the image to a tensor (this mimics ToTensor from torchvision)
    image = np.array(image).astype(np.float32) / 255.0  # Convert to float32 and normalize to [0, 1]
    image = torch.tensor(image).permute(2, 0, 1)  # Convert from HWC to CHW format
    
    # Normalize the image (this mimics Normalize from torchvision)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    image = (image - mean.view(-1, 1, 1)) / std.view(-1, 1, 1)  # Normalize each channel
    
    return image

# Read images from the folder
image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith(('jpg', 'jpeg', 'png'))]
print(f'Found {len(image_paths)} images in {image_folder}')

# Iterate over images in the folder
for image_path in image_paths:
    # Load and preprocess the image
    img = Image.open(image_path)
    img = transform_image(img)  # Apply the manual transformation
    
    # Add batch dimension
    img = img.unsqueeze(0)
    
    # Get predictions from the model
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = model(img)
        
        # Apply softmax to the predictions to get normalized probabilities
        probabilities = F.softmax(predictions, dim=1)
        
        # Get the maximum probability and the corresponding predicted class
        max_pred, argmax_pred = probabilities.max(dim=1)

    # Get the predicted label and confidence score
    pred = argmax_pred.item()
    confidence_score = max_pred.item()
    
    # Get the predicted label name from the class mapping
    pred_label_name = class_mapping.get(pred, str(pred))
    
    # Save the image and prediction
    save_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_pred_{pred_label_name}_conf_{confidence_score:.2f}.png")
    img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert to numpy for plotting
    plt.imshow(img_np)
    plt.title(f"Predicted: {pred_label_name}, Confidence: {confidence_score:.2f}")
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()
    
    # Store the results
    inputs_list.append(img)
    pred_list.append(pred)
    confidence_score_list.append(confidence_score)

# Save predictions and results to CSV
with open('image_predictions.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Filename', 'Predicted Label', 'Confidence Score'])
    for idx, (image_path, pred, score) in enumerate(zip(image_paths, pred_list, confidence_score_list)):
        pred_label_name = class_mapping.get(pred, str(pred))
        writer.writerow([os.path.basename(image_path), pred_label_name, score])

print("Images and CSV file saved successfully with predictions.")
