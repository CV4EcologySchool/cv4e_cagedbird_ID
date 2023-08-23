# I want to make it on the validation data
# has to be passed into the model how the dataloader does

import numpy as np
import comet_ml
from comet_ml import Experiment

import torch
import yaml

from train import create_dataloader, load_model, experiment # experiment should add the confusion matrix to the cometML experiment    # NOTE: since we're using these functions across files, it could make sense to put them in e.g. a "util.py" script.
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
dl_val = create_dataloader(cfg, split='val') # Or it could be with test, and then it should be labelled dl_test, this should in theory


# load 128 images (1 batch) across 29 of the classes

# load model - should load the saved model at the last checkpoint
model, start_epoch = load_model(cfg)

# Generate example true labels and predicted labels, start it for a few examples
inputs, labels = next(iter(dl_val))

for inputs, labels in dl_val:
    # Create NumPy arrays for the inputs and the labels

    # print ("The labels that are being loaded for the batch")
    # print (labels)

    # Compute predictions
    predictions = model(inputs) 
    predictions = predictions.argmax(dim=1).numpy() # argmax is saying what is the index position for the largest value in a list of numbers
    # I have to choose a number to label thi sample, hoose the inde  positin that has the highest score
    inputs = inputs.numpy()
    # print ("Print the shape of the inputs")
    # print (inputs)
    print(inputs.shape)
    labels = labels.numpy()

# Calculate the confusion matrix using scikit-learn
cm = confusion_matrix(labels, predictions)

# the colours need to mean something - need to scale the roles
# Plot confusion matrix to CometML
experiment.log_confusion_matrix(matrix=cm, images=inputs, title="Confusion Matrix", labels=labels.unique) # Try and plot the labels for each of the, add labels=labels (this will print),
# we just want the labels that are there in our batch, so it should be labels=labels.unique - this will show the unique labels for the batch
# 29 classes, which are currently mapped onto numbers as well; labels[0:28]

# Logs the image corresponding to the model prediction
experiment.log_confusion_matrix(
    y_test,
    predictions,
    images=x_test,
    title="Confusion Matrix: Evaluation",
    file_name="confusion-matrix-eval.json",
)

# if __name__ == '__main__':
#     # This block only gets executed if you call the "testing_metrics.py" script directly
#     # (i.e., "python ct_classifier/testing_metrics.py").
#     main()