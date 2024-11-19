import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
import pickle
import csv
import torch.nn.functional as F
from train import create_dataloader, load_model
from util import *

# Parameters
# Load the config for whichever is the best performing model
config = '/home/home01/bssbf/cv4e_cagedbird_ID/all_model_states/a-resnet18_d-checked_b-128_n-50_padded_images_sharpness_medium_more_data_upsampled/config_a-resnet18_d-checked_b-128_n-50_padded_images_sharpness_medium_more_data_upsampled.yaml'
split = 'test'


print(f'Using config "{config}"')
cfg = yaml.safe_load(open(config, 'r'))
init_seed(cfg.get('seed', None))

# Setup dataloader
dl_test = create_dataloader(cfg, split='test')

# Load model
model, start_epoch = load_model(cfg, load_latest_version=True)
print(start_epoch)