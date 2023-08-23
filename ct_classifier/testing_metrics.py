# I want to make it on the validation data
# has to be passed into the model how the dataloader does

import numpy as np
import matplotlib.pyplot as plt
import comet_ml
from comet_ml import Experiment

import torch
import yaml

from train import create_dataloader, load_model # experiment should add the confusion matrix to the cometML experiment    # NOTE: since we're using these functions across files, it could make sense to put them in e.g. a "util.py" script.
from sklearn.metrics import confusion_matrix

from util import * # To import th init seed from the util.py file in the same folder named ct_classifier
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

# Append wraps your item into a list, so you end up with a list of lists [[],[],[],[],[]]
# Extend puts the list into a newer list, but putting it into brackets [.....]

# Create NumPy arrays for the inputs and the labels and predictions, outside of the for loop now we have generated our lists
inputs_list = np.array(inputs_list)
labels_list = np.array(labels_list)
pred_list = np.array (pred_list)

print (pred_list)

# Calculate the confusion matrix using scikit-learn
cm1 = confusion_matrix(labels_list, pred_list)

# the colours need to mean something - need to scale the roles
# Plot confusion matrix to CometML

experiment = Experiment(
    api_key="6D79SKeAIuSjteySwQwqx96nq",
    project_name="cagedbird-classifier"
)
experiment.set_name("a-resnet18_d-high_b-128_n-75_padded_images_confusion_matrix")

experiment.log_confusion_matrix(matrix=cm1, title="Confusion Matrix 1", labels=labels) # images=inputs,

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