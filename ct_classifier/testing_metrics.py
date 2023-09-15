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
from sklearn.metrics import confusion_matrix
import csv


from util import * # To import the init seed from the util.py file in the same folder named ct_classifier

# Parameters
config = 'configs/exp_resnet18.yaml'
split = 'val'

# load config
print(f'Using config "{config}"')
cfg = yaml.safe_load(open(config, 'r'))

# Load the existing experiment key from file, for the experiment that is initialised in the train.py file, so these metrics can 
# be added to that experiment space in coment
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

# # Display a sample of images from the validation dataloader
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
    argmax_pred = predictions.argmax(dim=1) # argmax is saying what is the index position for the largest value in a list of number, so these
    # scores are the maximum prediction scores (i.e. the highest score so what will be the label reported for the class from the top-1 accuracy essentially)
    max_pred = predictions.max(dim=1).values
    print(max_pred.shape)
    # I have to choose a number to label this sample, it will choose the index position that has the highest score, we are convertin as we go so we don't need to relaebl
    # objects as they are converted anyway
    pred_list.extend(list(argmax_pred))
    max_pred_list.extend(list(max_pred))
    inputs_list.extend(list(inputs))
    labels_list.extend(list(labels))

# Experiment saving the predictions from the predictions list to a .csv file, 
# To save the full list of predictions as a CSV file, you iterate through the validation dataloader and make predictions using your model, appending each prediction to the pred_list:

with open('validation_predictions.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Prediction'])  # Write a header row if needed
    
    for prediction in pred_list:
        writer.writerow([prediction.item()])  # Write each prediction to a new row

# This code will help you see that the number of inputs, labels and lists match the number of predictions
# print ("Print the length of the inputs list")
# print(len(inputs_list))
# print ("Print the length of the labels list")
# print(len(labels_list))
# print ("Print the length of the max predictions list")
# print(len(max_pred_list))

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
plt.savefig('Histogram Scores Whole Model.png', dpi = 600)

# Try out the plots per class
# Calculate the number of unique classes in your data
num_classes = len(np.unique(labels_list))

# Save the class_maping in a pickle file, load it earlier so you can map the hist
with open('ct_classifier/class_mapping.pickle', 'rb') as f:
    class_mapping = pickle.load(f)

# You can print the classes if you 
# print(class_mapping)

# Create histograms for each class, initialising a dictionary
class_histograms = {}

# for class_idx in range(num_classes):
#     class_scores = []
#     for pred, label in zip(pred_list, labels_list):
#         if label == class_idx:
#             class_scores.append(pred.item())
#     plt.figure(figsize=(10, 6))
#     plt.hist(class_scores, bins=50, alpha=0.5, color='blue', label=f'Class {class_idx} Scores')
#     plt.xlabel('Class Scores')
#     plt.ylabel('Frequency')
#     plt.title(f'Histogram of Class {class_idx} Scores')
#     plt.legend()
#     plt.savefig(f'Class_{class_idx}_Histogram.png', dpi=600)
# #     plt.close()

for class_idx in range(num_classes):
    class_scores = []
    class_name = class_mapping[class_idx]  # Get the class name from your class mapping pickle file, so the histograms are loaded per class with the correct name
    
    for pred, label in zip(pred_list, labels_list):
        if label == class_idx:
            class_scores.append(pred.item())
    
    plt.figure(figsize=(10, 6))
    plt.hist(class_scores, bins=50, alpha=0.5, color='blue', label=f'{class_name} Scores')
    plt.xlabel('Class Scores')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {class_name} Scores')  # Use the class name in the title
    plt.legend()
    plt.savefig(f'{class_name}_histogram.png', dpi=600)
    plt.close()


# Create histograms for each class with the softmax scores
for class_idx in range(num_classes):
    class_scores = [functional.softmax(pred.float(), dim=0)[class_idx].item() for pred, label in zip(pred_list, labels_list) if label == class_idx]
    class_name = class_mapping[class_idx]
    
    plt.figure(figsize=(10, 6))
    plt.hist(class_scores, bins=50, alpha=0.5, color='blue', label=f'{class_name} Scores')
    plt.xlabel('Class Scores')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {class_name} Scores')
    plt.legend()
    plt.savefig(f'{class_name}_histogram.png', dpi=600)
    plt.close()

# In this modified code, we use F.softmax(pred, dim=0) to apply the softmax function to each prediction pred, and then we select the softmax score corresponding to the class_idx to create the histograms. This will show the distribution of softmax scores for each class.

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


# Assuming you have defined labels_list, pred_list, inputs_list, and class_mapping

# Calculate the confusion matrix using scikit-learn
cm1 = confusion_matrix(labels_list, pred_list)
print(cm1)

threshold = 0.3  # You can adjust this threshold value

# Calculate high confusion classes
confusion_sums = cm1.sum(axis=1) - np.diag(cm1)
high_confusion_classes = np.where(confusion_sums > threshold)[0]
print(high_confusion_classes)

# Create a custom grid layout
num_rows = len(high_confusion_classes)
num_columns = 4  # Number of columns for the grid
fig_width = 15  # Adjust the width of the figure

# Calculate the number of images per row and per column
images_per_row = min(num_columns, len(high_confusion_classes))
images_per_col = -(-len(high_confusion_classes) // num_columns)  # Ceiling division


fig, axes = plt.subplots(nrows=images_per_col, ncols=images_per_row, figsize=(fig_width, fig_width / images_per_row * images_per_col))

for i, class_idx in enumerate(high_confusion_classes):
    class_samples = np.where((labels_list == class_idx) & (labels_list != pred_list))[0]
    if len(class_samples) >= images_per_row:
        sample_indices = np.random.choice(class_samples, size=images_per_row, replace=False)
    else:
        # If there are fewer than 3 samples, repeat the available samples
        sample_indices = class_samples

    for j, sample_idx in enumerate(sample_indices):
        ax = axes[i // images_per_row, j % images_per_row]
        image = inputs_list[sample_idx].transpose(1, 2, 0)
        true_label = class_mapping[labels_list[sample_idx]]
        pred_label = class_mapping[pred_list[sample_idx]]

        ax.imshow(image)

        # Display the true and predicted labels above the image
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=8, color='black', ha='center')

        ax.axis('off')  # Turn off axes

# Remove any empty subplots
for i in range(len(high_confusion_classes), images_per_col):
    for j in range(images_per_row):
        axes[i, j].axis('off')

plt.tight_layout()

plt.savefig('Sample Of Bad Classes for Upsampled Data.png', dpi=600)


labels = unique_names_label_list

import itertools

import json

with open('cm.json') as f:
    data = json.load(f)

# labels = data['labels']

# print(labels)

target_names =[
      "Eurasian_jay",
      "Eurasian_siskin",
      "Grey_Parrot",
      "Hoopoe",
      "bluethroat",
      "bm_leafbird",
      "bnoriole",
      "bw_myna",
      "cf_white_eye",
      "chestnut_munia",
      "common_myna",
      "common_redpoll",
      "crested_lark",
      "fischers_lovebird",
      "great_myna",
      "hill_mynas",
      "hwamei",
      "japanese_grosbeak",
      "javan_pied_starling",
      "marsh_tit",
      "oriental_magpie_robin",
      "oriental_skylark",
      "red_billed_starling",
      "red whiskered_bulbul",
      "siberian_rubythroat",
      "swinhoes_whiteeye",
      "yellow_bellied_tits",
      "zebra dove",
      "zebra_finch"
    ]

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Oranges):#was Blues before, refer here for help: https://stackoverflow.com/questions/57043260/how-change-the-color-of-boxes-in-confusion-matrix-using-sklearn

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize=(20,20)) #was (10,10) before



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        cm = np.around(cm, decimals=2)

        cm[np.isnan(cm)] = 0.0

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
    
plot_confusion_matrix(cm1, target_names, title='Confusion Matrix')

# how to plot it: https://stackoverflow.com/questions/65317685/how-to-create-image-of-confusion-matrix-in-python
plt.savefig("cm_upsampled.png", dpi=500) # dpi can control the resolution

existing_experiment.log_confusion_matrix(matrix=cm1, title="Confusion Matrix 1", labels=unique_names_label_list) # images=inputs,

# Save confusion matrix plot 
# existing_experiment.log_image(plt, name='confusion_matrix.png')

existing_experiment.end()
