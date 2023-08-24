# I want to make it on the validation data
# has to be passed into the model how the dataloader does

import numpy as np
import matplotlib.pyplot as plt
import comet_ml
from comet_ml import Experiment, ExistingExperiment

import torch
import yaml

from train import create_dataloader, load_model # experiment should add the confusion matrix to the cometML experiment    # NOTE: since we're using these functions across files, it could make sense to put them in e.g. a "util.py" script.
from sklearn.metrics import confusion_matrix, average_precision_score, precision_recall_curve

from util import * # To import th init seed from the util.py file in the same folder named ct_classifier

# Load the experiment key from the file
with open("experiment_key.txt", "r") as file:
    experiment_key = file.read().strip()


# Provide the experiment key to continue an existing experiment
#  experiment_key = "b0ccf310a6d1443b8e89160a74f4680f"
existing_experiment = comet_ml.ExistingExperiment(api_key="6D79SKeAIuSjteySwQwqx96nq", previous_experiment=experiment_key)


# Parameters
config = 'configs/exp_resnet18.yaml'
split = 'val'

# load config
print(f'Using config "{config}"')
cfg = yaml.safe_load(open(config, 'r'))

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

# Here the internal Dataloader in PyTorch base code, knows that when it gets a DataLoader class that it will iterate
# through each item in the list rather than just generate a list for one batch in dl_val, so when we call dl_val
# we can iterate over all batches whereas this will just do one batch from the validation dataset, as loaded by the dataloader
# inputs, labels = next(iter(dl_val))

for inputs, labels in dl_val:
    predictions = model(inputs) 
    predictions = predictions.argmax(dim=1) # argmax is saying what is the index position for the largest value in a list of number
    # I have to choose a number to label this sample, it will choose the index position that has the highest score, we are convertin as we go so we don't need to relaebl
    # objects as they are converted anyway
    pred_list.extend(list(predictions))
    inputs_list.extend(list(inputs))
    labels_list.extend(list(labels))

print (labels)

# Append wraps your item into a list, so you end up with a list of lists [[],[],[],[],[]]
# Extend puts the list into a newer list, but putting it into brackets [.....]

# Create NumPy arrays for the inputs and the labels and predictions, outside of the for loop now we have generated our lists
inputs_list = np.array(inputs_list)
labels_list = np.array(labels_list)
pred_list = np.array (pred_list)

print (labels_list)

# print (pred_list)

# For each class, for a multi-label or multi-class situation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

# precision = dict()
# recall = dict()
# n_classes = 29
# average_precision = dict()
# for i in range(n_classes):
#     precision[i], recall[i], _ = precision_recall_curve(labels_list[:, i], pred_list[:, i])
#     average_precision[i] = average_precision_score(labels_list[:, i], pred_list[:, i])

# # A "micro-average": quantifying score on all classes jointly
# precision["micro"], recall["micro"], _ = precision_recall_curve(
#     labels_list.ravel(), pred_list.ravel()
# )
# average_precision["micro"] = average_precision_score(labels_list, pred_list, average="micro")

# print ("Print the average precision score")
# print (average_precision["micro"])
# print (average_precision_score)

# Calculate the confusion matrix using scikit-learn
cm1 = confusion_matrix(labels_list, pred_list)

# the colours need to mean something - need to scale the roles
# Plot confusion matrix to CometML

# use_experiment.py

# use_experiment.py


# Use the experiment key to interact with Comet.ml or perform any other action

# Experiment Key: f53b5ec49e694758827fd1d1d978ed05

# experiment = Experiment(
#     api_key="6D79SKeAIuSjteySwQwqx96nq",
#     project_name="cagedbird-classifier"
# )
# experiment.set_name("a-resnet18_d-high_b-128_n-75_padded_images_confusion_matrix")

# Map the class names to the labels

# # Sample data containing class information
# class_data = [
# {'id': 0, 'name': 'Eurasian_jay', 'supercategory': 'object'}
# {'id': 1, 'name': 'Eurasian_siskin', 'supercategory': 'object'}
# {'id': 2, 'name': 'Grey_Parrot', 'supercategory': 'object'}
# {'id': 3, 'name': 'Hoopoe', 'supercategory': 'object'}
# {'id': 4, 'name': 'bluethroat', 'supercategory': 'object'}
# {'id': 5, 'name': 'bm_leafbird', 'supercategory': 'object'}
# {'id': 6, 'name': 'bw_myna', 'supercategory': 'object'}
# {'id': 7, 'name': 'cf_white_eye', 'supercategory': 'object'}
# {'id': 8, 'name': 'chestnut_munia', 'supercategory': 'object'}
# {'id': 9, 'name': 'coal_tit_crops', 'supercategory': 'object'}
# {'id': 10, 'name': 'common_myna', 'supercategory': 'object'}
# {'id': 11, 'name': 'common_redpoll', 'supercategory': 'object'}
# {'id': 12, 'name': 'crested_lark', 'supercategory': 'object'}
# {'id': 13, 'name': 'fischers_lovebird', 'supercategory': 'object'}
# {'id': 14, 'name': 'great_myna', 'supercategory': 'object'}
# {'id': 15, 'name': 'hill_mynas', 'supercategory': 'object'}
# {'id': 16, 'name': 'hwamei', 'supercategory': 'object'}
# {'id': 17, 'name': 'japanese_grosbeak', 'supercategory': 'object'}
# {'id': 18, 'name': 'javan_pied_starling', 'supercategory': 'object'}
# {'id': 19, 'name': 'marsh_tit', 'supercategory': 'object'}
# {'id': 20, 'name': 'oriental_magpie_robin', 'supercategory': 'object'}
# {'id': 21, 'name': 'oriental_skylark', 'supercategory': 'object'}
# {'id': 22, 'name': 'red_billed_starling', 'supercategory': 'object'}
# {'id': 23, 'name': 'red_whiskered_bulbul', 'supercategory': 'object'}
# {'id': 24, 'name': 'siberian_rubythroat', 'supercategory': 'object'}
# {'id': 25, 'name': 'swinhoes_whiteeye', 'supercategory': 'object'}
# {'id': 26, 'name': 'yellow_bellied_tits', 'supercategory': 'object'}
# {'id': 27, 'name': 'zebra_dove', 'supercategory': 'object'}
# {'id': 28, 'name': 'zebra_finch', 'supercategory': 'object'}
# {'id': 0, 'name': 'Eurasian_jay', 'supercategory': 'object'}
# {'id': 1, 'name': 'Eurasian_siskin', 'supercategory': 'object'}
# {'id': 2, 'name': 'Grey_Parrot', 'supercategory': 'object'}
# {'id': 3, 'name': 'Hoopoe', 'supercategory': 'object'}
# {'id': 4, 'name': 'bluethroat', 'supercategory': 'object'}
# {'id': 5, 'name': 'bm_leafbird', 'supercategory': 'object'}
# {'id': 6, 'name': 'bw_myna', 'supercategory': 'object'}
# {'id': 7, 'name': 'cf_white_eye', 'supercategory': 'object'}
# {'id': 8, 'name': 'chestnut_munia', 'supercategory': 'object'}
# {'id': 9, 'name': 'coal_tit_crops', 'supercategory': 'object'}
# {'id': 10, 'name': 'common_myna', 'supercategory': 'object'}
# {'id': 11, 'name': 'common_redpoll', 'supercategory': 'object'}
# {'id': 12, 'name': 'crested_lark', 'supercategory': 'object'}
# {'id': 13, 'name': 'fischers_lovebird', 'supercategory': 'object'}
# {'id': 14, 'name': 'great_myna', 'supercategory': 'object'}
# {'id': 15, 'name': 'hill_mynas', 'supercategory': 'object'}
# {'id': 16, 'name': 'hwamei', 'supercategory': 'object'}
# {'id': 17, 'name': 'japanese_grosbeak', 'supercategory': 'object'}
# {'id': 18, 'name': 'javan_pied_starling', 'supercategory': 'object'}
# {'id': 19, 'name': 'marsh_tit', 'supercategory': 'object'}
# {'id': 20, 'name': 'oriental_magpie_robin', 'supercategory': 'object'}
# {'id': 21, 'name': 'oriental_skylark', 'supercategory': 'object'}
# {'id': 22, 'name': 'red_billed_starling', 'supercategory': 'object'}
# {'id': 23, 'name': 'red_whiskered_bulbul', 'supercategory': 'object'}
# {'id': 24, 'name': 'siberian_rubythroat', 'supercategory': 'object'}
# {'id': 25, 'name': 'swinhoes_whiteeye', 'supercategory': 'object'}
# {'id': 26, 'name': 'yellow_bellied_tits', 'supercategory': 'object'}
# {'id': 27, 'name': 'zebra_dove', 'supercategory': 'object'}
# {'id': 28, 'name': 'zebra_finch', 'supercategory': 'object'}
# ]

# # Create a mapping from 'id' to 'name'
# id_to_name_mapping = {class_info['id']: class_info['name'] for class_info in class_data}

# # # Sample numeric labels as tensors, but this should just be my labels_list
# # numeric_labels = [28, 0, 1, 10]  # Replace with your actual numeric labels

# # Map numeric labels to class names using the mapping
# class_names = [id_to_name_mapping[labels_list] for labels_list in labels_list]

# print("Mapped Class Names:", class_names)

existing_experiment.log_confusion_matrix(matrix=cm1, title="Confusion Matrix 1", labels=labels) # images=inputs,
existing_experiment.end()

# you should just be able to log the confusion matrix again with another line with a diffe
# AttributeError: 'numpy.ndarray' object has no attribute 'unique'
# Try and plot the labels for each of the, add labels=labels (this will print),
# we just want the labels that are there in our batch, so it should be labels=labels.unique - this will show the unique labels for the batch
# 29 classes, which are currently mapped onto numbers as well; labels[0:28]

# Start a new context for the second confusion matrix, we can use this if we are adding a plot to the same matrix, how do we add context to an experiment that
# is already connected to another one 
# with experiment.new_context("Confusion Matrix 2"):

# cm2 = confusion_matrix(labels1, predictions1)# Your second confusion matrix data
# experiment.log_confusion_matrix(matrix=cm2, title="Confusion Matrix 2")

# if __name__ == '__main__':
#     # This block only gets executed if you call the "testing_metrics.py" script directly
#     # (i.e., "python ct_classifier/testing_metrics.py").
#     main()


# histograms

# import torch

# # Model predictions
# probs = model(test_images)  

# # Ground truth labels
# test_labels = ... 

# # Get index of correct class 
# correct_class = test_labels.argmax(dim=1)

# # Get predicted probabilities for correct class
# predicted_probs = probs[torch.arange(len(probs)), correct_class]

# plt.hist(predicted_probs.numpy(), bins=20)


# other cometML code to show the images with the confusion_matrix, but we want to show an example of how to log a confusion matrix with images using CometML,
#     # by loading the whole batch


#     from comet_ml import Experiment
# import matplotlib.pyplot as plt

# # Start Comet experiment
# experiment = Experiment(project_name="confusion-matrix")

# # Get predictions and images
# preds = model.predict(x_test)
# images = x_test 

# # Calculate confusion matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, preds)

# # Plot confusion matrix
# plt.figure()
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title("Confusion Matrix")
# plt.colorbar()
# plt.tight_layout()

# # Log confusion matrix plot
# experiment.log_figure(figure=plt, figure_name="confusion-matrix.png") 

# # Log some example images 
# for i in range(5):
#   experiment.log_image(images[i], name="image"+str(i), image_channels="first")