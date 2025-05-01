import os
import torch
import KE_model
import importlib
from utils import net_utils
# from utils import csv_utils
# from layers import conv_type
from utils import path_utils
from configs.base_config import Config
import pdb
import wandb
import random
import numpy as np
import pathlib
from copy import deepcopy
import pickle


def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")

    return trainer.train, trainer.validate


def train_dense(cfg, generation, model=None, fisher_mat=None):

    if model is None:
        model = net_utils.get_model(cfg)
        if cfg.use_pretrain:
            net_utils.load_pretrained(cfg.init_path,cfg.gpu, model,cfg)


    if cfg.pretrained and cfg.pretrained != 'imagenet':
        net_utils.load_pretrained(cfg.pretrained,cfg.gpu, model,cfg)
        model = net_utils.move_model_to_gpu(cfg, model)
        if not cfg.no_reset:
            net_utils.split_reinitialize(cfg,model,reset_hypothesis=cfg.reset_hypothesis)
           
    model = net_utils.move_model_to_gpu(cfg, model)
    #save model immediately after init
    if cfg.save_model:
        run_base_dir, ckpt_base_dir, log_base_dir = path_utils.get_directories(cfg, generation)
        net_utils.save_checkpoint(
            {
                "epoch": 0,
                "arch": cfg.arch,
                "state_dict": model.state_dict(),
                # "optimizer": optimizer.state_dict(),
            },
            is_best=False,
            filename=ckpt_base_dir / f"init_model.state",
            save=False
        )
    
    cfg.trainer = 'default_cls'
    cfg.pretrained = None
    if cfg.reset_important_weights:
        ckpt_path, fisher_mat, model= KE_model.ke_cls_train_fish(cfg, model, generation, fisher_mat)
        
    else:

        ckpt_base_dir, model= KE_model.ke_cls_train(cfg, model, generation)
        sparse_mask = None
    non_overlapping_sparsemask = net_utils.create_dense_mask_0(deepcopy(model), cfg.device, value=0)
    if cfg.reset_important_weights:
       
        if cfg.snip :
            sparse_mask = None
            # check snip effectiveness
            sparse_model = net_utils.extract_sparse_weights(cfg, model, fisher_mat )
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
        tst_acc1, tst_acc5= KE_model.ke_cls_eval_sparse(cfg, model, generation, ckpt_path, 'acc_drop_reinit.csv')

        if cfg.freeze_fisher:
           model = net_utils.diff_lr_sparse(cfg, model, sparse_mask)
           print('freezing the important parameters')

    

    return model, fisher_mat, sparse_mask



def percentage_overlap(prev_mask, curr_mask, percent_flag=False):
    total_percent = {}
    for (name,prev_parm_m) , curr_parm_m in zip(prev_mask.named_parameters(),curr_mask.parameters()):
        if 'weight' in name and 'bn' not in name and 'downsample' not in name:

            overlap_param= ((prev_parm_m==curr_parm_m)*curr_parm_m).sum()
            # print(overlap_param)
            assert torch.numel(prev_parm_m) == torch.numel(curr_parm_m)
            N = torch.numel(prev_parm_m.data)
            if percent_flag == True:
                no_of_params = ((curr_parm_m==1)*1).sum()
                percent = overlap_param/no_of_params
            else:
                percent = overlap_param/N

            total_percent[name] = (percent *100)

    return total_percent


def start_KE(cfg):
    base_dir = pathlib.Path(f"{path_utils.get_checkpoint_dir()}/{cfg.name}")
    
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    ckpt_queue = []
    model=None
    fish_mat = None
      
    for gen in range(cfg.num_generations):
        cfg.start_epoch = 0
        model, fish_mat, sparse_mask  = train_dense(cfg, gen, model=model, fisher_mat=fish_mat)
        
        if cfg.num_generations == 1:
            break

def clean_dir(ckpt_dir,num_epochs):
    # print(ckpt_dir)
    if '0000' in str(ckpt_dir): ## Always keep the first model -- Help reproduce results
        return
    rm_path = ckpt_dir / 'model_best.pth'
    if rm_path.exists():
        os.remove(rm_path)

    rm_path = ckpt_dir / 'epoch_{}.state'.format(num_epochs - 1)
    if rm_path.exists():
        os.remove(rm_path)

    rm_path = ckpt_dir / 'initial.state'
    if rm_path.exists():
        os.remove(rm_path)

if __name__ == '__main__':
    cfg = Config().parse(None)
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not cfg.no_wandb:
        if len(cfg.group_vars) > 0:
            if len(cfg.group_vars) == 1:
                group_name = cfg.group_vars[0] + str(getattr(cfg, cfg.group_vars[0]))
            else:
                group_name = cfg.group_vars[0] + str(getattr(cfg, cfg.group_vars[0]))
                for var in cfg.group_vars[1:]:
                    group_name = group_name + '_' + var + str(getattr(cfg, var))
            wandb.init(project="llf_ke",
                   group=cfg.group_name,
                   name=group_name)
            for var in cfg.group_vars:
                wandb.config.update({var:getattr(cfg, var)})
                
    if cfg.seed is not None and cfg.fix_seed: #FIXING SEED LEADS TO SAME REINITIALIZATION VALUES FOR EACH GENERATION
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)              

    start_KE(cfg)
