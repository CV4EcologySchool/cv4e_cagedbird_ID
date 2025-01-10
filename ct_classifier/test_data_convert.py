import os
import shutil
import csv
import pickle

# Paths
original_test_folder = '/home/home01/bssbf/cv4e_cagedbird_ID/test'  # Path to the original test set folder with species subfolders
flattened_test_folder = '/home/home01/bssbf/cv4e_cagedbird_ID/test_con2'  # Path where the flattened images will be stored
true_labels_csv = 'true_labels_test5.csv'  # CSV file for true labels
class_mapping_file = '/home/home01/bssbf/cv4e_cagedbird_ID/ct_classifier/class_mapping_56.pickle'  # Path to your class mapping pickle file

# Load the class mapping pickle file
with open(class_mapping_file, 'rb') as f:
    class_mapping = pickle.load(f)

# Invert the class_mapping to map class names to their indices
inverted_class_mapping = {v: k for k, v in class_mapping.items()}

# Create the flattened test folder if it doesn't exist
if not os.path.exists(flattened_test_folder):
    os.makedirs(flattened_test_folder)

# Initialize list to store true labels
true_labels = []

# Iterate through each species subfolder in the original test folder
for species_folder in os.listdir(original_test_folder):
    species_folder_path = os.path.join(original_test_folder, species_folder)
    
    if os.path.isdir(species_folder_path):  # Process only directories
        print(f"Processing folder: {species_folder}")
        
        # Get the class index corresponding to the species name
        class_index = inverted_class_mapping.get(species_folder, None)
        if class_index is not None:
            print(f"Found class index {class_index} for species folder {species_folder}")
            
            # Get all image files in this species folder
            for img_file in os.listdir(species_folder_path):
                img_path = os.path.join(species_folder_path, img_file)
                if os.path.isfile(img_path):  # Check if it's an image file
                    # Move the image to the new flattened test folder
                    new_img_path = os.path.join(flattened_test_folder, img_file)
                    shutil.copy(img_path, new_img_path)  # Copy image to the flattened folder
                    
                    # Prepare the true label as the species class index
                    true_labels.append([img_file, class_index])  # Filename, True label (class index)
        else:
            print(f"Warning: No class index found for folder '{species_folder}'")

# Write the true labels to a CSV
with open(true_labels_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Filename', 'True Label'])  # Header
    writer.writerows(true_labels)

print(f"True labels have been saved to {true_labels_csv}")
print(f"Images have been flattened to {flattened_test_folder}")
