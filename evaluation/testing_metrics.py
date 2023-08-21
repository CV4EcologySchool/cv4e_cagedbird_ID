# I want to make it on the validation data
# has to be passed into the model how the dataloader does

import numpy as np
import torch
import yaml
from ct_classifier.train import create_dataloader, load_model       # NOTE: since we're using these functions across files, it could make sense to put them in e.g. a "util.py" script.
from sklearn.metrics import confusion_matrix
from comet_ml import Experiment

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

# load model
model = load_model(cfg)

# Generate example true labels and predicted labels, start it for a few examples
inputs, labels = next(iter(dl_val))

# Create NumPy arrays for the inputs and the labels
inputs = inputs.numpy()
labels = labels.numpy()

print ("The labels that are being loaded for the batch")
print (labels)

breakpoint

experiment = comet_ml.Experiment(
    api_key="6D79SKeAIuSjteySwQwqx96nq",
    project_name="cagedbird-classifier"
)
experiment.set_name ("a-resnet18_d-high_b-128_n-35")

# Compute predictions
predictions = model(inputs)  
predictions = predictions.argmax(dim=1).numpy()

# Calculate the confusion matrix using scikit-learn
cm = confusion_matrix(labels, predictions)

# Plot confusion matrix to CometML
experiment.log_confusion_matrix(matrix=cm, title="Confusion Matrix") # Try and plot the labels for each of the, add labels=label
# 29 classes, which are currently mapped onto numbers as well

if __name__ == '__main__':
    # This block only gets executed if you call the "testing_metrics.py" script directly
    # (i.e., "python ct_classifier/testing_metrics.py").
    main()