import os
import sys
import torch
import KE_model
import importlib
from utils import net_utils, path_utils
from configs.base_config import Config
import wandb
import torch.nn as nn
import random
import numpy as np
import pathlib
from copy import deepcopy
import csv
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.logging import AverageMeter, ProgressMeter
from utils.eval_utils import accuracy
from layers.CS_KD import KDLoss
from torch.utils.tensorboard import SummaryWriter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to get training and validation functions from the specified trainer module
def get_trainer(cfg):
    trainer = importlib.import_module(f"trainers.{cfg.trainer}")
    return trainer.train, trainer.validate

# Function to train the model for a single generation
def train_dense(cfg, generation, model=None, fisher_mat=None):
    if model is None:
        model = net_utils.get_model(cfg)
        if cfg.use_pretrain:
            net_utils.load_pretrained(cfg.init_path, cfg.gpu, model, cfg)

    if cfg.pretrained and cfg.pretrained != 'imagenet':
        net_utils.load_pretrained(cfg.pretrained, cfg.gpu, model, cfg)
        model = net_utils.move_model_to_gpu(cfg, model)
        if not cfg.no_reset:
            net_utils.split_reinitialize(cfg, model, reset_hypothesis=cfg.reset_hypothesis)

    model = net_utils.move_model_to_gpu(cfg, model)

    if cfg.save_model:
        run_base_dir, ckpt_base_dir, log_base_dir = path_utils.get_directories(cfg, generation)
        net_utils.save_checkpoint(
            {"epoch": 0, "arch": cfg.arch, "state_dict": model.state_dict()},
            is_best=False,
            filename=ckpt_base_dir / f"init_model.state",
            save=False
        )

    cfg.trainer = 'default_cls'
    cfg.pretrained = None

    if cfg.reset_important_weights:
        if cfg.prune_criterion in ["SNIP", "SNIPit", "SNAPit"  , "CNIPit"]:
            ckpt_path, fisher_mat, model, epoch_metrics = KE_model.ke_cls_train_fish(cfg, model, generation, fisher_mat)
            sparse_model = net_utils.extract_sparse_weights(cfg, model, fisher_mat)
            tst_acc1, tst_acc5 = KE_model.ke_cls_eval_sparse(cfg, sparse_model, generation, ckpt_path, 'acc_pruned_model.csv')
            model = net_utils.reparameterize_non_sparse(cfg, model, fisher_mat)
            sparse_mask = fisher_mat
            torch.save(sparse_mask.state_dict(), os.path.join(cfg.exp_dir, f"{cfg.prune_criterion.lower()}_mask_{generation}.pth"))
            tst_acc1_reinit, tst_acc5_reinit = KE_model.ke_cls_eval_sparse(cfg, model, generation, ckpt_path, 'acc_drop_reinit.csv')
            epoch_metrics['test_acc1'][-1] = tst_acc1_reinit
            epoch_metrics['test_acc5'][-1] = tst_acc5_reinit
        else:
            ckpt_path, fisher_mat, model, epoch_metrics = KE_model.ke_cls_train_fish(cfg, model, generation, fisher_mat)
            sparse_mask, dict_FIM = net_utils.extract_new_sparse_model(cfg, model, fisher_mat, generation)
            torch.save(sparse_mask.state_dict(), os.path.join(cfg.exp_dir, f"sparse_mask_{generation}.pth"))
            np.save(os.path.join(cfg.exp_dir, f"FIM_{generation}.npy"), fisher_mat.cpu().detach().numpy())
            model = net_utils.reparameterize_non_sparse(cfg, model, sparse_mask)
            tst_acc1, tst_acc5 = KE_model.ke_cls_eval_sparse(cfg, model, generation, ckpt_path, 'acc_drop_reinit.csv')
            epoch_metrics['test_acc1'][-1] = tst_acc1
            epoch_metrics['test_acc5'][-1] = tst_acc5
    else:
        ckpt_base_dir, model, epoch_metrics = KE_model.ke_cls_train(cfg, model, generation)
        sparse_mask = net_utils.create_dense_mask_0(deepcopy(model), cfg.device, value=1)

    non_overlapping_sparsemask = net_utils.create_dense_mask_0(deepcopy(model), cfg.device, value=0)

    return model, fisher_mat, sparse_mask, epoch_metrics

# Function to calculate the percentage overlap between previous and current masks
def percentage_overlap(prev_mask, curr_mask, percent_flag=False):
    total_percent = {}
    for (name, prev_parm_m), curr_parm_m in zip(prev_mask.named_parameters(), curr_mask.parameters()):
        if 'weight' in name and 'bn' not in name and 'downsample' not in name:
            prev_parm_m_np = prev_parm_m.detach().cpu().numpy()
            curr_parm_m_np = curr_parm_m.detach().cpu().numpy()
            overlap_param = np.sum((prev_parm_m_np == curr_parm_m_np) * curr_parm_m_np)
            N = prev_parm_m_np.size
            if percent_flag:
                no_of_params = np.sum((curr_parm_m_np == 1) * 1)
                percent = overlap_param / no_of_params if no_of_params > 0 else 0
            else:
                percent = overlap_param / N
            total_percent[name] = percent * 100
    return total_percent

# Main function to start the Knowledge Evolution process
def start_KE(cfg):
    base_dir = pathlib.Path(f"{path_utils.get_checkpoint_dir()}/{cfg.name}")
    os.makedirs(base_dir, exist_ok=True)

    ckpt_queue = []
    model = None
    fish_mat = None

    weights_history = {
        'conv1': [], 'layer1.0.conv1': [], 'layer2.0.conv1': [],
        'layer3.0.conv1': [], 'layer4.0.conv1': [], 'fc': []
    }
    mask_history = {}
    all_epoch_data = []

    for gen in range(cfg.num_generations):
        cfg.start_epoch = 0
        model, fish_mat, sparse_mask, epoch_metrics = train_dense(cfg, gen, model=model, fisher_mat=fish_mat)

        # Store weights
        weights_history['conv1'].append(model.conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history['layer1.0.conv1'].append(model.layer1[0].conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history['layer2.0.conv1'].append(model.layer2[0].conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history['layer3.0.conv1'].append(model.layer3[0].conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history['layer4.0.conv1'].append(model.layer4[0].conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history['fc'].append(model.fc.weight.data.clone().cpu().numpy().flatten())

        # Store masks
        mask_history[gen] = {}
        if sparse_mask is not None:
            for name, param in sparse_mask.named_parameters():
                if 'weight' in name and 'bn' not in name and 'downsample' not in name:
                    mask_history[gen][name] = param.data.clone().cpu().numpy()

        # Add epoch data
        try:
            expected_length = cfg.epochs
            for key in ['train_acc1', 'train_acc5', 'train_loss', 'test_acc1', 'test_acc5', 'test_loss', 'avg_sparsity', 'mask_update']:
                actual_length = len(epoch_metrics[key])
                if actual_length != expected_length:
                    epoch_metrics[key].extend([None] * (expected_length - actual_length))
            for layer in epoch_metrics['layer_sparsity']:
                actual_length = len(epoch_metrics['layer_sparsity'][layer])
                if actual_length != expected_length:
                    epoch_metrics['layer_sparsity'][layer].extend([0] * (expected_length - actual_length))

            epoch_df = pd.DataFrame({
                'Epoch': range(cfg.epochs),
                'Generation': [gen] * cfg.epochs,
                'Train_Acc@1': epoch_metrics['train_acc1'],
                'Train_Acc@5': epoch_metrics['train_acc5'],
                'Train_Loss': epoch_metrics['train_loss'],
                'Test_Acc@1': epoch_metrics['test_acc1'],
                'Test_Acc@5': epoch_metrics['test_acc5'],
                'Test_Loss': epoch_metrics['test_loss'],
                'Avg_Sparsity': epoch_metrics['avg_sparsity'],
                'Mask_Update': epoch_metrics['mask_update']
            })
            all_epoch_data.append(epoch_df)
        except Exception as e:
            logger.error(f"Failed to create DataFrame for generation {gen}: {str(e)}")
            epoch_df = pd.DataFrame({
                'Epoch': range(cfg.epochs),
                'Generation': [gen] * cfg.epochs,
                'Train_Acc@1': [None] * cfg.epochs,
                'Train_Acc@5': [None] * cfg.epochs,
                'Train_Loss': [None] * cfg.epochs,
                'Test_Acc@1': [None] * cfg.epochs,
                'Test_Acc@5': [None] * cfg.epochs,
                'Test_Loss': [None] * cfg.epochs,
                'Avg_Sparsity': [None] * cfg.epochs,
                'Mask_Update': [False] * cfg.epochs
            })
            all_epoch_data.append(epoch_df)

        if cfg.num_generations == 1:
            break

    # Combine all epoch data
    try:
        df = pd.concat(all_epoch_data, ignore_index=True)
    except Exception as e:
        logger.error(f"Failed to concatenate DataFrames: {str(e)}")
        df = pd.DataFrame()

    # Plotting
    if not df.empty:
        # Accuracy Plot (Seaborn)
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Epoch', y='Train_Acc@1', hue='Generation', data=df, color='blue')
        sns.lineplot(x='Epoch', y='Test_Acc@1', hue='Generation', data=df, color='orange', marker='o')
        for epoch in df[df['Mask_Update']]['Epoch'].unique():
            plt.axvline(x=epoch, color='red', linestyle='--', alpha=0.5)
        plt.title(f'Train and Test Accuracy Over Epochs {cfg.set}, {cfg.arch}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend(['Train Acc@1', 'Test Acc@1', 'Mask Update'])
        plt.grid(True)
        plt.savefig(os.path.join(base_dir, 'accuracy_over_epochs.png'))
        plt.close()

        # Loss Plot (Plotly)
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for idx, gen in enumerate(df['Generation'].unique()):
            gen_df = df[df['Generation'] == gen]
            color = colors[idx % len(colors)]
            fig.add_trace(go.Scatter(
                x=gen_df['Epoch'],
                y=gen_df['Train_Loss'],
                mode='lines',
                name=f'Train Loss Gen {gen}',
                line=dict(color=color, width=2)
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=gen_df['Epoch'],
                y=gen_df['Test_Loss'],
                mode='lines+markers',
                name=f'Test Loss Gen {gen}',
                line=dict(color=color, width=2, dash='dash'),
                marker=dict(size=8)
            ), row=1, col=1)
        for epoch in df[df['Mask_Update']]['Epoch'].unique():
            fig.add_vline(x=epoch, line=dict(color='red', dash='dash', width=1), opacity=0.5)
        fig.update_layout(
            title=f'Train and Test Loss Over Epochs ({cfg.set}, {cfg.arch})',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            showlegend=True,
            template='plotly_white',
            legend=dict(x=1.05, y=1, xanchor='left', yanchor='top', bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='Black', borderwidth=1)
        )
        fig.write_html(os.path.join(base_dir, 'loss_over_epochs.html'))
        fig.write_image(os.path.join(base_dir, 'loss_over_epochs.png'))

        # Sparsity Plot (Seaborn)
        sparsity_data = []
        for gen in mask_history:
            for layer_name, mask in mask_history[gen].items():
                sparsity = 100 * (1 - mask.mean())
                sparsity_data.append({'Generation': gen, 'Layer': layer_name, 'Sparsity': sparsity})
        sparsity_df = pd.DataFrame(sparsity_data)
        plt.figure(figsize=(14, 6))
        sns.barplot(x='Layer', y='Sparsity', hue='Generation', data=sparsity_df)
        plt.title(f'Sparsity Across Layers at Mask Update Points {cfg.set}, {cfg.arch}')
        plt.xlabel('Layer')
        plt.ylabel('Sparsity (%)')
        plt.xticks(rotation=45)
        plt.legend(title='Generation')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, 'sparsity_across_layers.png'))
        plt.close()

        # Layer-wise Sparsity Plot (Seaborn, for selected layers over epochs)
        selected_layers = ['conv1.weight', 'layer1.0.conv1.weight', 'layer2.0.conv1.weight', 'layer3.0.conv1.weight', 'layer4.0.conv1.weight', 'fc.weight']
        for gen in range(cfg.num_generations):
            layer_sparsity_data = []
            for epoch in range(cfg.epochs):
                for layer in selected_layers:
                    if layer in epoch_metrics['layer_sparsity']:
                        sparsity = epoch_metrics['layer_sparsity'][layer][epoch]
                        layer_sparsity_data.append({'Epoch': epoch, 'Layer': layer, 'Sparsity': sparsity})
            layer_sparsity_df = pd.DataFrame(layer_sparsity_data)
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='Epoch', y='Sparsity', hue='Layer', data=layer_sparsity_df, marker='o')
            for epoch in df[df['Mask_Update']]['Epoch'].unique():
                plt.axvline(x=epoch, color='red', linestyle='--', alpha=0.5)
            plt.title(f'Sparsity Over Epochs for Generation {gen} {cfg.set}, {cfg.arch}')
            plt.xlabel('Epoch')
            plt.ylabel('Sparsity (%)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(base_dir, f'layer_sparsity_over_epochs_gen_{gen}.png'))
            plt.close()

        # Mask Overlap Plot (Seaborn)
        overlap_data = []
        for gen1 in mask_history:
            for gen2 in mask_history:
                if gen1 < gen2:
                    prev_model = deepcopy(model)
                    curr_model = deepcopy(model)
                    for (name, param) in prev_model.named_parameters():
                        if name in mask_history[gen1]:
                            param.data = torch.from_numpy(mask_history[gen1][name]).to(param.device)
                    for (name, param) in curr_model.named_parameters():
                        if name in mask_history[gen2]:
                            param.data = torch.from_numpy(mask_history[gen2][name]).to(param.device)
                    overlap = percentage_overlap(prev_model, curr_model, percent_flag=True)
                    for layer, perc in overlap.items():
                        overlap_data.append({'Layer': layer, 'Comparison': f'Gen {gen1} vs Gen {gen2}', 'Overlap': perc})
        overlap_df = pd.DataFrame(overlap_data)
        if not overlap_df.empty:
            plt.figure(figsize=(14, 6))
            sns.barplot(x='Layer', y='Overlap', hue='Comparison', data=overlap_df)
            plt.title(f'Mask Overlap Between Different Generations ({cfg.set}, {cfg.arch})')
            plt.xlabel('Layer')
            plt.ylabel('Overlap (%)')
            plt.xticks(rotation=45)
            plt.legend(title='Comparison')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(base_dir, 'mask_overlap.png'))
            plt.close()

# Function to clean up checkpoint directory
def clean_dir(ckpt_dir, num_epochs):
    if '0000' in str(ckpt_dir):
        return
    for fname in ['model_best.pth', f'epoch_{num_epochs - 1}.state', 'initial.state']:
        rm_path = ckpt_dir / fname
        if rm_path.exists():
            os.remove(rm_path)

# Main execution block
if __name__ == '__main__':
    cfg = Config().parse(sys.argv[1:])  # Parse command-line arguments using Config

    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.conv_type = 'SplitConv'
    cfg.logger = logger

    if not cfg.no_wandb:
        if len(cfg.group_vars) > 0:
            group_name = cfg.group_vars[0] + str(getattr(cfg, cfg.group_vars[0]))
            for var in cfg.group_vars[1:]:
                group_name += '_' + var + str(getattr(cfg, var))
            wandb.init(project="llf_ke", group=cfg.group_name, name=group_name)
            for var in cfg.group_vars:
                wandb.config.update({var: getattr(cfg, var)})

    if cfg.seed is not None and cfg.fix_seed:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    start_KE(cfg)