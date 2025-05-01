# utils/constants.py

# Directory for saving checkpoints
checkpoints_dir = './checkpoints'

# Directory for datasets
datasets_dir = './data/datasets'

# Ensure these directories exist
import os
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(datasets_dir, exist_ok=True)
