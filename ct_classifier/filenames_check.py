import numpy as np
import matplotlib.pyplot as plt
import torch
import csv
from sklearn.metrics import confusion_matrix, average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

# Assuming the rest of your code is the same

# Initialize lists to store inputs, labels, and predictions
inputs_list = []
labels_list = []
pred_list = []
max_pred_list = []

# Iterate through the validation data
for inputs, labels in dl_val:
    predictions = model(inputs)
    argmax_pred = predictions.argmax(dim=1)
    max_pred = predictions.max(dim=1).values

    # Append data to respective lists
    pred_list.extend(list(argmax_pred))
    max_pred_list.extend(list(max_pred))
    inputs_list.extend(list(inputs))
    labels_list.extend(list(labels))

# Convert lists to numpy arrays
inputs_list = np.array(inputs_list)
labels_list = np.array(labels_list)
pred_list = np.array(pred_list)

# Get class mapping (if you haven't loaded it yet)
with open('ct_classifier/class_mapping.pickle', 'rb') as f:
    class_mapping = pickle.load(f)

# Find indices where predictions do not match the true labels
incorrect_indices = np.where(labels_list != pred_list)[0]

# Sample a few of these incorrect predictions (adjust the number if necessary)
sample_size = 12  # Adjust based on how many images you want to display
sampled_indices = np.random.choice(incorrect_indices, size=sample_size, replace=False)

# Set up a grid for displaying images
fig, axes = plt.subplots(3, 4, figsize=(12, 8))  # Adjust the grid size as needed
axes = axes.flatten()

# Display images with their true and predicted labels
for idx, ax in zip(sampled_indices, axes):
    image = inputs_list[idx].transpose(1, 2, 0)  # Convert from CHW to HWC for display
    true_label = class_mapping[labels_list[idx]]
    pred_label = class_mapping[pred_list[idx]]

    # Display the image
    ax.imshow(image)
    ax.set_title(f'True: {true_label}\nPred: {pred_label}')
    ax.axis('off')  # Turn off axis labels

plt.tight_layout()
plt.show()

# Optionally save the figure
plt.savefig('true_vs_predicted_samples.png', dpi=600)

# For confusion matrix and evaluation metrics (AUPRC, etc.)
# Calculate confusion matrix
cm1 = confusion_matrix(labels_list, pred_list)

# Calculate average precision score (AUPRC)
one_hot_labels = label_binarize(labels_list, classes=list(range(len(np.unique(labels_list)))))
one_hot_preds = label_binarize(pred_list, classes=list(range(len(np.unique(pred_list)))))
auprc = average_precision_score(one_hot_labels, one_hot_preds, average=None)

# Print confusion matrix and AUPRC
print("Confusion Matrix:")
print(cm1)
print("Average Precision Score (AUPRC):")
print(auprc)
