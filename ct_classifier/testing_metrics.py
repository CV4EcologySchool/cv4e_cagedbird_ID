# I want to make it on the validation data
# has to be passed into the model how the dataloader does

import numpy as np
import matplotlib.pyplot as plt
import comet_ml
from comet_ml import Experiment, ExistingExperiment

import torch
import yaml
import pickle 
import json

from train import create_dataloader, load_model # experiment should add the confusion matrix to the cometML experiment    # NOTE: since we're using these functions across files, it could make sense to put them in e.g. a "util.py" script.
from sklearn.metrics import confusion_matrix, average_precision_score, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import torch.nn.functional as functional  # Import functional

from util import * # To import the init seed from the util.py file in the same folder named ct_classifier

# Parameters
config = 'configs/exp_resnet18.yaml'
split = 'val'

# load config
print(f'Using config "{config}"')
cfg = yaml.safe_load(open(config, 'r'))

# Load the experiment key from the file
with open("experiment_key.txt", "r") as file:
    experiment_key = file.read().strip()

# Provide the experiment key to continue an existing experiment
existing_experiment = comet_ml.ExistingExperiment(api_key=cfg["api_key"], previous_experiment=experiment_key)

# Set the seed, so you can reproduce the randomness, None is there as null because the seed is already in the config
init_seed(cfg.get('seed', None))

# setup entities
dl_val = create_dataloader (cfg, split='val') # Or it could be with test, and then it should be labelled dl_test, this should in theory
# load 128 images (1 batch) across 29 of the classes, for all of the batches within the dataloader

# load model - should load the saved model at the last checkpoint
model, start_epoch = load_model(cfg, load_latest_version=True)
print(start_epoch)

# # Display the images
# fig1 = plt.figure(figsize=(12, 8))
# for idx in range(12):
#     ax1 = fig1.add_subplot(3, 4, idx + 1, xticks=[], yticks=[])
#     # The imshow function is used to display the images, and the loop displays a sample of 12 images along with their corresponding labels
#     ax1.imshow(inputs[idx].permute(1, 2, 0)) # or is to transpose?

# plt.tight_layout()
# # plt.show()
# plt.savefig("val_loader2.png")

inputs_list = []
labels_list = []
pred_list = []
max_pred_list = []

# Here the internal Dataloader in PyTorch base code, knows that when it gets a DataLoader class that it will iterate
# through each item in the list rather than just generate a list for one batch in dl_val, so when we call dl_val
# we can iterate over all batches whereas this will just do one batch from the validation dataset, as loaded by the dataloader

for inputs, labels in dl_val:
    predictions = model(inputs) 
    argmax_pred = predictions.argmax(dim=1) # argmax is saying what is the index position for the largest value in a list of number
    max_pred = predictions.max(dim=1).values
    print(max_pred.shape)
    # I have to choose a number to label this sample, it will choose the index position that has the highest score, we are convertin as we go so we don't need to relaebl
    # objects as they are converted anyway
    pred_list.extend(list(argmax_pred))
    max_pred_list.extend(list(max_pred))
    inputs_list.extend(list(inputs))
    labels_list.extend(list(labels))

print ("Print the length of the inputs list")
print(len(inputs_list))
print ("Print the length of the labels list")
print(len(labels_list))
print ("Print the length of the max predictions list")
print(len(max_pred_list))

# Calculate the number of unique classes in your data
num_classes = len(np.unique(labels_list))

# Create histograms for positive and negative class scores
positive_scores = []
negative_scores = []

for i in range(len(labels_list)):
    pred_score = max_pred_list [i].detach().numpy()  # Detach and then convert to NumPy
    if labels_list[i] == pred_list[i]:
        positive_scores.append(pred_score)
    else:
        negative_scores.append(pred_score)

# Plot histograms
num_bins = 50  # You can adjust this value based on your preference
plt.figure(figsize=(10, 6))
plt.hist(positive_scores, bins=num_bins, alpha=0.5, color='blue', label='Positive Class Scores')
plt.hist(negative_scores, bins=num_bins, alpha=0.5, color='red', label='Negative Class Scores')
plt.xlabel('Class Scores')
plt.ylabel('Frequency')
plt.title('Histogram of Positive and Negative Class Scores')
plt.legend()
# plt.show()
plt.savefig('Histogram Scores')

# # Try out the plots per class
# # Calculate the number of unique classes in your data
# num_classes = len(np.unique(labels_list))

# # Initialize lists to hold scores for each class
# class_logit_scores = [[] for _ in range(num_classes)]
# class_softmax_scores = [[] for _ in range(num_classes)]

# for i in range(len(labels_list)):
#     logit_scores = max_pred_list[i].detach().cpu().numpy()
#     softmax_scores = functional.softmax(max_pred_list[i], dim=0).detach().cpu().numpy()
#     class_label = labels_list[i].item()
#     class_logit_scores[class_label].extend([logit_scores])  # Extend logit_scores as a list
    
#     # Convert softmax_scores to a list only if it's a tensor with multiple elements
#     if softmax_scores.ndim > 0:
#         class_softmax_scores[class_label].extend(softmax_scores.tolist())

# # Create subplots for histograms of logit scores
# num_rows = (num_classes + 2) // 3
# fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
# fig.subplots_adjust(hspace=0.5)

# for class_label, scores in enumerate(class_logit_scores):
#     row = class_label // 3
#     col = class_label % 3
#     ax = axes[row, col]
    
#     ax.hist(scores, bins=num_bins, alpha=0.5)  # Plot logit_scores array
        
#     ax.set_title(f'Class {class_label}')
#     ax.set_xlabel('Logit Scores')
#     ax.set_ylabel('Frequency')

# # Save the logit scores plot
# plt.tight_layout()
# plt.savefig('Logit_Scores_Per_Class_Panel.png')

# # Create subplots for histograms of softmax scores
# fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
# fig.subplots_adjust(hspace=0.5)

# for class_label, scores in enumerate(class_softmax_scores):
#     row = class_label // 3
#     col = class_label % 3
#     ax = axes[row, col]
    
#     ax.hist(scores, bins=num_bins, alpha=0.5)  # Plot softmax_scores array
        
#     ax.set_title(f'Class {class_label}')
#     ax.set_xlabel('Softmax Scores')
#     ax.set_ylabel('Frequency')

# # Save the softmax scores plot
# plt.tight_layout()
# plt.savefig('Softmax_Scores_Per_Class_Panel.png')

# Append wraps your item into a list, so you end up with a list of lists [[],[],[],[],[]]
# Extend puts the list into a newer list, but putting it into brackets [.....]

# Create NumPy arrays for the inputs and the labels and predictions, outside of the for loop now we have generated our lists
inputs_list = np.array(inputs_list)
labels_list = np.array(labels_list)
pred_list = np.array (pred_list)

print("Print the length of the label list")
print(len(labels_list))

print("Print the length of the pred list")
print (len(pred_list))

# Use label_binarize to be multi-label like settings
one_hot_labels = label_binarize(labels_list, classes=list(range(len(np.unique(labels_list))))) # this will index from 0-28 for 29 classes
n_classes = one_hot_labels.shape[1]

one_hot_preds = label_binarize(pred_list, classes=list(range(len(np.unique(pred_list))))) # this will index from 0-28 for 29 classes
n_classes = one_hot_preds.shape[1]

# auprc = average_precision_score(labels.detach().cpu().numpy() , prediction.detach().cpu().numpy(),average=None)
auprc = average_precision_score(one_hot_labels, one_hot_preds,average=None)
print ("The average precision score on the validation data")
print (auprc)

# Save the class_maping in a pickle file
with open('ct_classifier/class_mapping.pickle', 'rb') as f:
    class_mapping = pickle.load(f)

print(class_mapping)

# Intialise a list to store the names to get the unique lists
names_label_list = []

for label in labels_list:
    name = class_mapping [label] # each "0" and then that 0 in every labe to 'Eurasian jay'
    names_label_list.append(name)

unique_names_label_list = []

for i in range (cfg['num_classes']): # change from range (29) to range(cfg['num_classes'])
    name = class_mapping [i]
    unique_names_label_list.append(name)

names_pred_list = []
for predictions in pred_list:
    name = class_mapping [predictions] # each "0" and then that 0 in every labe to 'Eurasian jay'
    names_pred_list.append(name)

for i in range (cfg['num_classes']):
    name = class_mapping [i]
    unique_names_label_list.append(name)

# print (names_label_list)

# Calculate the confusion matrix using scikit-learn
cm1 = confusion_matrix(labels_list, pred_list)

threshold = 0.3  # You can adjust this threshold value

# Calculate high confusion classes
confusion_sums = cm1.sum(axis=1) - np.diag(cm1)
high_confusion_classes = np.where(confusion_sums > threshold)[0]

# # Sample 9 random images 
# random_images = np.random.choice(np.where((true_labels[idx] in high_confusion_classes) & 
#                                           (true_labels[idx] != pred_labels[idx])), size=9, replace=False)

# # Plot
# fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8), 
#                          gridspec_kw={'wspace': 0.5, 'hspace': 0.5})

# for i, idx in enumerate(random_images):
#     ax = axes[i//3, i%3]
#     image = images[idx] 
#     true_label = true_labels[idx]
#     pred_label = pred_labels[idx]
    
#     ax.imshow(image, cmap='gray')
#     ax.set_title(f"True: {true_label}\nPred: {pred_label}")
#     ax.axis('off')
# plt.savefig('Sample Of Bad Classes.png')

labels = unique_names_label_list

plt.imshow(cm1, interpolation='nearest')
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.xticks(range(len(labels)), labels)  
plt.yticks(range(len(labels)), labels)

# Save confusion matrix plot 
plt.savefig('confusion_matrix.png')

existing_experiment.log_confusion_matrix(matrix=cm1, title="Confusion Matrix 1", labels=unique_names_label_list) # images=inputs,

# Save confusion matrix plot 
# existing_experiment.log_image(plt, name='confusion_matrix.png')

existing_experiment.end()
