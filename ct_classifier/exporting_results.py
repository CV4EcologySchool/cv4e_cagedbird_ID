import json
import torch
import pickle
from util import *  # Ensure your utility functions are imported

# Load the class mapping (if required for class names)
with open('ct_classifier/class_mapping.pickle', 'rb') as f:
    class_mapping = pickle.load(f)

# Function to capture annotations from the validation dataloader
def capture_annotations(dl_val):
    annotations = []
    for inputs, labels in dl_val:
        for idx, label in enumerate(labels):
            # Assuming each 'inputs' contains image data and you want to capture filename, true label, etc.
            annotation = {
                'filename': f'image_{idx}.jpg',  # Replace with actual filename if you have one
                'true_label': class_mapping[label.item()],  # Map true label to class name
                'prediction': None,  # You can leave prediction blank for now
                'max_prediction': None  # You can leave max_prediction blank for now
            }
            annotations.append(annotation)
    return annotations

# Capture annotations from the validation dataloader
annotations = capture_annotations(dl_val)

# Export the captured annotations to a new JSON file
with open('captured_annotations.json', 'w') as f:
    json.dump(annotations, f, indent=4)

# Now, you can process predictions and update the annotations in the JSON as needed
for inputs, labels in dl_val:
    predictions = model(inputs)
    argmax_pred = predictions.argmax(dim=1)
    max_pred = predictions.max(dim=1).values

    # Update the annotations with predictions and max prediction values
    for idx, annotation in enumerate(annotations):
        annotation['prediction'] = class_mapping[argmax_pred[idx].item()]
        annotation['max_prediction'] = max_pred[idx].item()

# Export the final annotations with predictions to a new JSON file
with open('annotations_with_predictions.json', 'w') as f:
    json.dump(annotations, f, indent=4)
