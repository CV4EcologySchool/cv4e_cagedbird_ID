'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    2022 Benjamin Kellenberger
'''

import os
import argparse
import yaml
import glob
from tqdm import trange

# COMET WARNING: To get all data logged automatically, import comet_ml before the following modules: torch.
import comet_ml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD

# let's import our own classes and functions!
from util import init_seed
from dataset import CTDataset
from model import CustomResNet18
import matplotlib.pyplot as plt
import numpy as np

def create_dataloader(cfg, split='train'):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = CTDataset(cfg, split)        # create an object instance of our CTDataset class
    print ("Print the length of the dataset")
    print(len(dataset_instance))
    
    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['num_workers']
        )
    return dataLoader



def load_model(cfg):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = CustomResNet18(cfg['num_classes'])         # create an object instance of our CustomResNet18 class

    # load latest model state
    model_states = glob.glob('model_states/*.pt')
    model_states = [] # Sets it to 0, and not see any checkpoint files: Hey this has been changed during training since we are not ready to resume
    if len(model_states):
        # at least one save state found; get latest
        model_epochs = [int(m.replace('model_states/','').replace('.pt','')) for m in model_states]
        start_epoch = max(model_epochs) 
        # load state dict and apply weights to model
        print(f'Resuming from epoch {start_epoch}')
        state = torch.load(open(f'model_states/{start_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])

    else:
        # no save state found; start anew
        print('Starting new model')
        start_epoch = 0

    return model_instance, start_epoch



def save_model(cfg, epoch, model, stats):
    # make sure save directory exists; create if not
    os.makedirs('model_states', exist_ok=True)

    # get model parameters and add to stats...
    stats['model'] = model.state_dict()

    # ...and save
    torch.save(stats, open(f'model_states/{epoch}.pt', 'wb'))
    
    # also save config file if not present
    cfpath = 'model_states/config.yaml'
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)

def setup_optimizer(cfg, model):
    '''
        The optimizer is what applies the gradients to the parameters and makes
        the model learn on the dataset.
    '''
    optimizer = SGD(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])
    return optimizer



def train(cfg, dataLoader, model, optimizer):
    '''
        Our actual training function.
    '''

    device = cfg['device']

    # put model on device
    model.to(device)
    
    # put the model into training mode
    # this is required for some layers that behave differently during training
    # and validation (examples: Batch Normalization, Dropout, etc.)
    model.train()

    # needs to have a softmax function : # Apply softmax to convert logits to probabilities
    # softmax = nn.Softmax(dim=1)
    # probabilities = softmax(logits)
    # print(probabilities)

    # loss function
    criterion = nn.CrossEntropyLoss() # the softmax is internal to this function: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

    # running averages
    loss_total, oa_total = 0.0, 0.0                         # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    for idx, (data, labels) in enumerate(dataLoader):       # see the last line of file "dataset.py" where we return the image tensor (data) and label

        # put data and labels on device
        data, labels = data.to(device), labels.to(device)

        # forward pass
        prediction = model(data)

        # reset gradients to zero
        optimizer.zero_grad()

        # loss
        loss = criterion(prediction, labels)

        # backward pass (calculate gradients of current batch)
        loss.backward()

        # apply gradients to model parameters
        optimizer.step()

        # log statistics
        loss_total += loss.item()                       # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor

        pred_label = torch.argmax(prediction, dim=1)    # the predicted label is the one at position (class index) with highest predicted value
        oa = torch.mean((pred_label == labels).float()) # OA: number of correct predictions divided by batch size (i.e., average/mean)
        oa_total += oa.item()

        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(idx+1),
                100*oa_total/(idx+1)
            )
        )
        progressBar.update(1)
    
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)           # shorthand notation for: loss_total = loss_total / len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total



def validate(cfg, dataLoader, model):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()
    
    criterion = nn.CrossEntropyLoss()   # we still need a criterion to calculate the validation loss

    # running averages
    loss_total, oa_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    
    with torch.no_grad():               # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels) in enumerate(dataLoader):

            # put data and labels on device
            data, labels = data.to(device), labels.to(device)

            # forward pass
            prediction = model(data)

            # loss
            loss = criterion(prediction, labels)

            # log statistics
            loss_total += loss.item()

            pred_label = torch.argmax(prediction, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()

            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)
    
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total



def main():

    # For Comet to start tracking a training run,
# just add these two lines at the top of
# your training script:

    experiment = comet_ml.Experiment(
        api_key="6D79SKeAIuSjteySwQwqx96nq",
        project_name="cagedbird-classifier"
    )
    experiment.set_name("a-resnet18_d-high_b-128_n-50_padded_images")

    # architecture name: 
    # dataset type:_high
    # batch size:
    # number of epochs: 
    resume = False # to update an existing experiment... or not

    if resume:
        experiment = comet_ml.ExistingExperiment(
            api_key="6D79SKeAIuSjteySwQwqx96nq",
            experiment_key="cfac36ee909a4b3b8417991e522f3423",
        )

    else:
        experiment = comet_ml.Experiment(
            api_key="6D79SKeAIuSjteySwQwqx96nq",
            project_name="cagedbird-classifier",
        )

# your model training or evaluation code

# Metrics from this training run will now be
# available in the Comet UI

    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/exp_resnet18.yaml
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet18.yaml')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))
     
    # this is the yaml one loaded as cfg
    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'

    # initialize data loaders for training and validation set
    dl_train = create_dataloader(cfg, split='train')
    sample_batch = next(iter(dl_train))
    inputs, labels = sample_batch

    # Display the images
    fig = plt.figure(figsize=(12, 8))
    for idx in range(12):
        ax = fig.add_subplot(3, 4, idx + 1, xticks=[], yticks=[])
        # The imshow function is used to display the images, and the loop displays a sample of 12 images along with their corresponding labels
        ax.imshow(inputs[idx].permute(1, 2, 0)) # or is to transpose?

    plt.tight_layout()
    # plt.show()
    plt.savefig("output.png")


    # Add a debugging breakpoint here

    dl_val = create_dataloader(cfg, split='val')
    print ("Length of training dataloader")

    # Number of training samples divided by batch size
    print(len(dl_train))
    print ("Length of validation dataloader")
    print(len(dl_val))

    # initialize model
    model, current_epoch = load_model(cfg)

    # set up model optimizer
    optim = setup_optimizer(cfg, model)

    # we have everything now: data loaders, model, optimizer; let's do the epochs!
    numEpochs = cfg['num_epochs']
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        loss_train, oa_train = train(cfg, dl_train, model, optim)
        loss_val, oa_val = validate(cfg, dl_val, model)

        # combine stats and save
        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'oa_val': oa_val
        }

        experiment.log_metric("Training loss", loss_train, step=current_epoch) # could do batch later
        experiment.log_metric("Validation loss", loss_val, step=current_epoch) # could do batch later
        experiment.log_metric("Training accuracy", oa_train, step = current_epoch)
        experiment.log_metric("Validation accuracy", oa_val, step = current_epoch)

         # Log hyperparameters like the learning rate and the batch size
        for param_name, param_value in cfg.items():
            experiment.log_parameter(param_name, param_value)

        save_model(cfg, current_epoch, model, stats)
    
if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()

# CTDataset.__getitem__()
# CTDataset.len ()
# dataLoader.len()