# test_xception_structure.py
import sys
import torch
from configs.base_config import Config

sys.path.append('..')  # Adds the parent directory (DNR) to the system path
from models.split_resnet import Split_Xception  

# Default configuration setup
cfg = Config().parse(None)
cfg.arch = 'xception'
cfg.num_cls = 10  # For example, for CIFAR10
cfg.pretrained = 'imagenet'

# Model creation
model = Split_Xception(cfg)

# Printing the structure of the Xception model
print("Xception model structure:")
for name, module in model.model.named_modules():
    print(f"{name}: {module.__class__.__name__}")
