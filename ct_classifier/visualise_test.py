import yaml
from ct_classifier.train import create_dataloader, load_model       # NOTE: since we're using these functions across files, it could make sense to put them in e.g. a "util.py" script.

# load parameters and the test set
config = 'configs/exp_resnet18.yaml'
split = 'test'

# load config
print(f'Using config "{config}"')
cfg = yaml.safe_load(open(config, 'r'))


# setup entities
dl_test = create_dataloader(cfg, split='test')

# load model
model = load_model(cfg)