import os
import torch
import KE_model
import importlib
from utils import net_utils
from utils import path_utils
from configs.base_config import Config
import wandb
import random
import numpy as np
import pathlib
from copy import deepcopy
import pickle
import matplotlib.pyplot as plt

# Function to get training and validation functions from the specified trainer module
def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")
    return trainer.train, trainer.validate

# Function to train the model for a single generation
def train_dense(cfg, generation, model=None, fisher_mat=None):
    # If no model is provided, create a new one using the configuration
    if model is None:
        model = net_utils.get_model(cfg)
        # Load pretrained weights if specified
        if cfg.use_pretrain:
            net_utils.load_pretrained(cfg.init_path, cfg.gpu, model, cfg)

    # Load pretrained model if specified and not using ImageNet weights
    if cfg.pretrained and cfg.pretrained != 'imagenet':
        net_utils.load_pretrained(cfg.pretrained, cfg.gpu, model, cfg)
        model = net_utils.move_model_to_gpu(cfg, model)
        # Reinitialize weights if reset is enabled
        if not cfg.no_reset:
            net_utils.split_reinitialize(cfg, model, reset_hypothesis=cfg.reset_hypothesis)
    
    # Move model to GPU
    model = net_utils.move_model_to_gpu(cfg, model)
    
    # Save initial model state if saving is enabled
    if cfg.save_model:
        run_base_dir, ckpt_base_dir, log_base_dir = path_utils.get_directories(cfg, generation)
        net_utils.save_checkpoint(
            {
                "epoch": 0,
                "arch": cfg.arch,
                "state_dict": model.state_dict(),
            },
            is_best=False,
            filename=ckpt_base_dir / f"init_model.state",
            save=False
        )
    
    # Set trainer and reset pretrained flag
    cfg.trainer = 'default_cls'
    cfg.pretrained = None
    
    # Train using either fisher-based or standard method based on config
    if cfg.reset_important_weights:
        ckpt_path, fisher_mat, model = KE_model.ke_cls_train_fish(cfg, model, generation, fisher_mat)
    else:
        ckpt_base_dir, model = KE_model.ke_cls_train(cfg, model, generation)
        sparse_mask = None
    
    # Create a mask with all zeros for non-overlapping parameters
    non_overlapping_sparsemask = net_utils.create_dense_mask_0(deepcopy(model), cfg.device, value=0)
    
    # Handle sparsity and reinitialization if reset_important_weights is enabled
    if cfg.reset_important_weights:
        if cfg.snip:
            sparse_mask = None
            sparse_model = net_utils.extract_sparse_weights(cfg, model, fisher_mat)
            print('sparse model acc')
            tst_acc1, tst_acc5 = KE_model.ke_cls_eval_sparse(cfg, sparse_model, generation, ckpt_path, 'acc_pruned_model.csv')
            model = net_utils.reparameterize_non_sparse(cfg, model, fisher_mat)
            torch.save(fisher_mat.state_dict(), os.path.join(base_pth, "snip_mask_{}.pth".format(generation)))
            print('resetting non important params based on snip for next generation')
        else:
            sparse_mask = net_utils.extract_new_sparse_model(cfg, model, fisher_mat, generation)
            torch.save(sparse_mask.state_dict(), os.path.join(base_pth, "sparse_mask_{}.pth".format(generation)))
            np.save(os.path.join(base_pth, "FIM_{}.npy".format(generation), fisher_mat.cpu().detach().numpy()))
            model = net_utils.reparameterize_non_sparse(cfg, model, sparse_mask)
        tst_acc1, tst_acc5 = KE_model.ke_cls_eval_sparse(cfg, model, generation, ckpt_path, 'acc_drop_reinit.csv')

        if cfg.freeze_fisher:
            model = net_utils.diff_lr_sparse(cfg, model, sparse_mask)
            print('freezing the important parameters')

    return model, fisher_mat, sparse_mask

# Function to calculate the percentage overlap between previous and current masks
def percentage_overlap(prev_mask, curr_mask, percent_flag=False):
    total_percent = {}
    for (name, prev_parm_m), curr_parm_m in zip(prev_mask.named_parameters(), curr_mask.parameters()):
        if 'weight' in name and 'bn' not in name and 'downsample' not in name:
            overlap_param = ((prev_parm_m == curr_parm_m) * curr_parm_m).sum()
            assert torch.numel(prev_parm_m) == torch.numel(curr_parm_m)
            N = torch.numel(prev_parm_m.data)
            if percent_flag:
                no_of_params = ((curr_parm_m == 1) * 1).sum()
                percent = overlap_param / no_of_params
            else:
                percent = overlap_param / N
            total_percent[name] = (percent * 100)
    return total_percent

# Main function to start the Knowledge Evolution process
def start_KE(cfg):
    # Set base directory for saving results
    base_dir = pathlib.Path(f"{path_utils.get_checkpoint_dir()}/{cfg.name}")
    
    # Create directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    ckpt_queue = []
    model = None
    fish_mat = None
    
    # Dictionary to store weights history for multiple layers
    weights_history = {
        'conv1': [],
        'layer1.0.conv1': [],
        'layer2.0.conv1': [],
        'layer3.0.conv1': [],
        'layer4.0.conv1': []
    }
      
    # Iterate over the specified number of generations
    for gen in range(cfg.num_generations):
        cfg.start_epoch = 0
        # Train the model for the current generation
        model, fish_mat, sparse_mask = train_dense(cfg, gen, model=model, fisher_mat=fish_mat)
        
        # Store weights for selected layers
        weights_history['conv1'].append(model.conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history['layer1.0.conv1'].append(model.layer1[0].conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history['layer2.0.conv1'].append(model.layer2[0].conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history['layer3.0.conv1'].append(model.layer3[0].conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history['layer4.0.conv1'].append(model.layer4[0].conv1.weight.data.clone().cpu().numpy().flatten())
        
        # Break if only one generation is specified
        if cfg.num_generations == 1:
            break

    # Plot weights for each layer
    for layer_name, weights_list in weights_history.items():
        plt.figure(figsize=(12, 5))
        for gen, weights in enumerate(weights_list):
            plt.plot(weights[:10], label=f'Generation {gen}', alpha=0.7)  # Plot only first 10 weights for readability
        plt.title(f"Changes in {layer_name} Weights Across Generations")
        plt.xlabel("Weight Index")
        plt.ylabel("Weight Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(base_dir, f"{layer_name}_weights_plot.png"))  # Save plot as PNG
        plt.show()  # Display plot

# Function to clean up checkpoint directory
def clean_dir(ckpt_dir, num_epochs):
    # Skip cleaning if directory contains '0000' (first model)
    if '0000' in str(ckpt_dir):
        return
    # Remove best model file if it exists
    rm_path = ckpt_dir / 'model_best.pth'
    if rm_path.exists():
        os.remove(rm_path)
    # Remove last epoch file if it exists
    rm_path = ckpt_dir / f'epoch_{num_epochs - 1}.state'
    if rm_path.exists():
        os.remove(rm_path)
    # Remove initial state file if it exists
    rm_path = ckpt_dir / 'initial.state'
    if rm_path.exists():
        os.remove(rm_path)

# Main execution block
if __name__ == '__main__':
    # Parse configuration
    cfg = Config().parse(None)
    # Set device to GPU if available
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Ensure SplitConv is used for convolutional layers
    cfg.conv_type = 'SplitConv'
    
    # Initialize Weights & Biases if enabled
    if not cfg.no_wandb:
        if len(cfg.group_vars) > 0:
            if len(cfg.group_vars) == 1:
                group_name = cfg.group_vars[0] + str(getattr(cfg, cfg.group_vars[0]))
            else:
                group_name = cfg.group_vars[0] + str(getattr(cfg, cfg.group_vars[0]))
                for var in cfg.group_vars[1:]:
                    group_name = group_name + '_' + var + str(getattr(cfg, var))
            wandb.init(project="llf_ke", group=cfg.group_name, name=group_name)
            for var in cfg.group_vars:
                wandb.config.update({var: getattr(cfg, var)})
                
    # Set random seeds if fixed seed is enabled
    if cfg.seed is not None and cfg.fix_seed:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
    
    # Start the Knowledge Evolution process
    start_KE(cfg)
