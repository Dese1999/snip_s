import os
import time
import data
import torch
import random
import importlib
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
from utils import net_utils
from utils import csv_utils
from utils import path_utils
# from layers import ml_losses
from datetime import timedelta
import torch.utils.data.distributed
from utils.schedulers import get_policy
from torch.utils.tensorboard import SummaryWriter
from utils.logging import AverageMeter, ProgressMeter
import wandb
import torch.nn.functional as F
from copy import deepcopy
import csv
from utils.pruning import Pruner
import matplotlib.pyplot as plt
import numpy as np
from utils.pruning import Pruner



def ke_cls_train(cfg, model, generation):
    cfg.logger.info(cfg)
    if cfg.seed is not None and cfg.fix_seed:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    train, validate = get_trainer(cfg)

    if cfg.gpu is not None:
        cfg.logger.info("Use GPU: {} for training".format(cfg.gpu))

    optimizer = get_optimizer(cfg, model)
    cfg.logger.info(f"=> Getting {cfg.set} dataset")
    dataset = getattr(data, cfg.set)(cfg)
    
    if cfg.lr_policy == 'long_cosine_lr':
        lr_policy = get_policy(cfg.lr_policy)(optimizer, generation, cfg)
    else:
        lr_policy = get_policy(cfg.lr_policy)(optimizer, cfg)

    if cfg.label_smoothing == 0:
        softmax_criterion = nn.CrossEntropyLoss().cuda()
    else:
        softmax_criterion = net_utils.LabelSmoothing(smoothing=cfg.label_smoothing).cuda()


    criterion = lambda output,target: softmax_criterion(output, target)


    # optionally resume from a checkpoint
    best_val_acc1 = 0.0
    best_val_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    if cfg.resume:
        best_val_acc1 = resume(cfg, model, optimizer)

    run_base_dir, ckpt_base_dir, log_base_dir = path_utils.get_directories(cfg, generation) 
    cfg.ckpt_base_dir = ckpt_base_dir

    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time],cfg, prefix="Overall Timing"
    )

    end_epoch = time.time()
    cfg.start_epoch = cfg.start_epoch or 0
    last_val_acc1 = None
    last_val_acc5 = None


    start_time = time.time()
    end_epochs = cfg.epochs
    if generation == 0 and cfg.use_pretrain:
        end_epochs = 0
    else:
        end_epochs = cfg.epochs
    # Start training
    bad_val_counter = 0
    
    for epoch in range(cfg.start_epoch, end_epochs):
        if cfg.lr_policy == "val_dependent_lr":
            if bad_val_counter >= 20:
                lr_policy(net_utils.get_lr(optimizer), 2.0)
                bad_val_counter = 0
            else:
                lr_policy(None, None)
        else:
            lr_policy(epoch, iteration=None)

        cur_lr = net_utils.get_lr(optimizer)
        if not cfg.no_wandb:
            wandb.log({'Epoch': epoch + (generation*end_epochs), 'LR': cur_lr})
        # print(cur_lr)
        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5 = train(
                dataset.train_loader, model, criterion, optimizer, epoch, cfg, writer=writer
            )
        train_time.update((time.time() - start_train) / 60)
      
        if not cfg.no_wandb:
            wandb.log({'Epoch': epoch + (generation*end_epochs), 'Generation':generation, 'Train Acc1': train_acc1, 'Train Acc5': train_acc5})

        if (epoch+1) % cfg.test_interval == 0:
            # evaluate on validation set
            start_validation = time.time()
            last_val_acc1, last_val_acc5 = validate(dataset.val_loader, model, criterion, cfg, writer, epoch)
            validation_time.update((time.time() - start_validation) / 60)

            if not cfg.no_wandb:
                wandb.log({'Epoch': epoch + (generation*end_epochs), 'Generation':generation, 'Val Acc1': last_val_acc1, 'Val Acc5': last_val_acc5})          
                
            # remember best acc@1 and save checkpoint
            is_best = last_val_acc1 > best_val_acc1
            best_val_acc1 = max(last_val_acc1, best_val_acc1)
            if last_val_acc1 * 0.999 < best_val_acc1:
                bad_val_counter += 1
            best_val_acc5 = max(last_val_acc5, best_val_acc5)
            best_train_acc1 = max(train_acc1, best_train_acc1)
            best_train_acc5 = max(train_acc5, best_train_acc5)

            save = (((epoch+1) % cfg.save_every) == 0) and cfg.save_every > 0

            elapsed_time = time.time() - start_time
            seconds_todo = (cfg.epochs - epoch) * (elapsed_time/cfg.test_interval)
            estimated_time_complete = timedelta(seconds=int(seconds_todo))
            start_time = time.time()
            
            if cfg.save_model:
                if is_best or save or epoch == cfg.epochs - 1:
                    if is_best:
                        cfg.logger.info(f"==> best {last_val_acc1:.02f} saving at {ckpt_base_dir / 'model_best.pth'}")

                    net_utils.save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "arch": cfg.arch,
                            "state_dict": model.state_dict(),
                            "best_acc1": best_val_acc1,
                            "best_acc5": best_val_acc5,
                            "best_train_acc1": best_train_acc1,
                            "best_train_acc5": best_train_acc5,
                            "optimizer": optimizer.state_dict(),
                            "curr_acc1": last_val_acc1,
                            "curr_acc5": last_val_acc5,
                        },
                        is_best,
                        filename=ckpt_base_dir / f"epoch_{epoch}.state",
                        save=save or epoch == cfg.epochs - 1,
                    )

            epoch_time.update((time.time() - end_epoch) / 60)
            progress_overall.display(epoch)
            progress_overall.write_to_tensorboard(
                writer, prefix="diagnostics", global_step=epoch
            )


            writer.add_scalar("test/lr", cur_lr, epoch)
            end_epoch = time.time()

        if cfg.eval_intermediate_tst>0 and cfg.eval_tst and (epoch+1) % cfg.eval_intermediate_tst == 0:
            last_tst_acc1, last_tst_acc5 = validate(dataset.tst_loader, model, criterion, cfg, writer, 0)
            best_tst_acc1 = 0
            best_tst_acc5 = 0
            if not cfg.no_wandb:
                wandb.log({'Epoch':epoch, 'Test Acc1': last_tst_acc1, 'Test Acc5': last_tst_acc5})
        else:
            last_tst_acc1 = 0
            last_tst_acc5 = 0
            best_tst_acc1 = 0
            best_tst_acc5 = 0


    if cfg.eval_tst and cfg.eval_intermediate_tst==0: #cfg.epochs < 300 and
        last_tst_acc1, last_tst_acc5 = validate(dataset.tst_loader, model, criterion, cfg, writer, 0)
        best_tst_acc1 = 0
        best_tst_acc5 = 0
        if not cfg.no_wandb:
            wandb.log({'Generation':generation, 'Test Acc1': last_tst_acc1, 'Test Acc5': last_tst_acc5})
    else:
        last_tst_acc1 = 0
        last_tst_acc5 = 0
        best_tst_acc1 = 0
        best_tst_acc5 = 0

    col_names = ['generation', 'sparcity', 'last_val_acc1', 'last_val_acc5', 'best_val_acc1', 'best_val_acc5',
                 'last_tst_acc1', 'last_tst_acc5', 'best_tst_acc1', 'best_tst_acc5', 'best_train_acc1',
                 'best_train_acc5']
    arg_list = [generation, cfg.sparsity, last_val_acc1, last_val_acc5, best_val_acc1, best_val_acc5, last_tst_acc1,
                last_tst_acc5, best_tst_acc1, best_tst_acc5, best_train_acc1, best_train_acc5]

    csv_file = os.path.join(run_base_dir, "train.csv")

    if not os.path.exists(csv_file):
        with open(csv_file, 'a') as ff:
            wr = csv.writer(ff, quoting=csv.QUOTE_ALL)
            wr.writerow(col_names)

    with open(csv_file, 'a') as ff:
        wr = csv.writer(ff, quoting=csv.QUOTE_ALL)
        wr.writerow(arg_list)
   

    cfg.logger.info(f"==> Final Best {best_val_acc1:.02f}, saving at {ckpt_base_dir / 'model_best.pth'}")

    
    return ckpt_base_dir, model

def fisher_matrix(cfg, net, dataset, opt, fisher_mat):

    fish = torch.zeros_like(net.get_params())
    logsoft = nn.LogSoftmax(dim=1)
    for j, data in enumerate(dataset):
        inputs, labels = data[0].to(cfg.device), data[1].long().squeeze().to(cfg.device)

        opt.zero_grad()
        output = net(inputs)
        loss = - F.nll_loss(logsoft(output), labels,
                            reduction='none')

        exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
        loss = torch.mean(loss)
        loss.backward()
        fish += exp_cond_prob * net.get_grads() ** 2

    fish /= (len(dataset) * cfg.batch_size)

    if fisher_mat is None:
        fisher_mat = fish
    else:
        # fisher_mat = cfg.gamma* fisher_mat + (1- cfg.gamma)* fish
        fisher_mat *= cfg.gamma
        fisher_mat += fish

    # checkpoint = net.get_params().data.clone()
    return fisher_mat

def get_trainer(args):
    args.logger.info(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.default_cls")
    return trainer.train, trainer.validate


def resume(args, model, optimizer):
    if os.path.isfile(args.resume):
        args.logger.info(f"=> Loading checkpoint '{args.resume}'")

        checkpoint = torch.load(args.resume, map_location=f"cuda:{args.gpu}")
        if args.start_epoch is None:
            args.logger.info(f"=> Setting new start epoch at {checkpoint['epoch']}")
            args.start_epoch = checkpoint["epoch"]

        best_acc1 = checkpoint["best_acc1"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        args.logger.info(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        return best_acc1
    else:
        args.logger.info(f"=> No checkpoint found at '{args.resume}'")




def get_optimizer(args, model,fine_tune=False,criterion=None):
    # for n, v in model.named_parameters():
    #     if v.requires_grad:
    #         args.logger.info("<DEBUG> gradient to {}".format(n))

    #     if not v.requires_grad:
    #         args.logger.info("<DEBUG> no gradient to {}".format(n))

    param_groups = model.parameters()
    if fine_tune:
        # Train Parameters
        param_groups = [
            {'params': list(
                set(model.parameters()).difference(set(model.model.embedding.parameters()))) if args.gpu != -1 else
            list(set(model.module.parameters()).difference(set(model.module.model.embedding.parameters())))},
            {
                'params': model.model.embedding.parameters() if args.gpu != -1 else model.module.model.embedding.parameters(),
                'lr': float(args.lr) * 1},
        ]
        if args.ml_loss == 'Proxy_Anchor':
            param_groups.append({'params': criterion.proxies, 'lr': float(args.lr) * 100})

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                          momentum=args.momentum, weight_decay=args.weight_decay)

    elif args.optimizer == "sgd_TEMP": #use this for freeze layer experiments, so there are no parameter updates for frozen layers
        parameters = list(model.named_parameters())
        param_groups = [v for n, v in parameters if v.requires_grad]        
        optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                          momentum=args.momentum, weight_decay=args.weight_decay)     
        
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, param_groups), lr=args.lr
        )
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(param_groups, lr=args.lr, alpha=0.9, weight_decay = args.weight_decay, momentum = 0.9)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay = args.weight_decay)
    else:
        raise NotImplemented('Invalid Optimizer {}'.format(args.optimizer))

    return optimizer
    ##############################################################################
def ke_cls_train_fish(cfg, model, generation, fisher_mat, teacher_model=None):
    if cfg.seed is not None and cfg.fix_seed:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    train, validate = get_trainer(cfg)

    if cfg.gpu is not None:
        cfg.logger.info("Use GPU: {} for training".format(cfg.gpu))

    optimizer = get_optimizer(cfg, model)
    cfg.logger.info(f"=> Getting {cfg.set} dataset")
    dataset = getattr(data, cfg.set)(cfg)

    if cfg.lr_policy == 'long_cosine_lr':
        lr_policy = get_policy(cfg.lr_policy)(optimizer, generation, cfg)
    else:
        lr_policy = get_policy(cfg.lr_policy)(optimizer, cfg)

    if cfg.label_smoothing == 0:
        softmax_criterion = nn.CrossEntropyLoss().cuda()
    else:
        softmax_criterion = net_utils.LabelSmoothing(smoothing=cfg.label_smoothing).cuda()

    criterion = lambda output, target: softmax_criterion(output, target)

    # Metrics collection
    epoch_metrics = {
        'train_acc1': [], 'train_acc5': [], 'train_loss': [],
        'test_acc1': [], 'test_acc5': [], 'test_loss': [],
        'avg_sparsity': [], 'mask_update': [],
        'layer_sparsity': {}  # Store sparsity per layer
    }

    best_tst_acc1 = 0.0
    best_tst_acc5 = 0.0
    best_val_acc1 = 0.0
    best_val_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    if cfg.resume:
        best_val_acc1 = resume(cfg, model, optimizer)

    run_base_dir, ckpt_base_dir, log_base_dir = path_utils.get_directories(cfg, generation)
    cfg.ckpt_base_dir = ckpt_base_dir

    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], cfg, prefix="Overall Timing"
    )

    end_epoch = time.time()
    cfg.start_epoch = cfg.start_epoch or 0
    last_val_acc1 = None
    last_val_acc5 = None

    start_time = time.time()
    end_epochs = cfg.epochs
    if generation == 0 and cfg.use_pretrain:
        end_epochs = 0

    # Initialize layer sparsity dictionary
    for name, param in model.named_parameters():
        if 'weight' in name and 'bn' not in name and 'downsample' not in name:
            epoch_metrics['layer_sparsity'][name] = []

    bad_val_counter = 0

    try:
        for epoch in range(cfg.start_epoch, end_epochs):
            if cfg.lr_policy == "val_dependent_lr":
                if bad_val_counter >= 20:
                    lr_policy(net_utils.get_lr(optimizer), 2.0)
                    bad_val_counter = 0
                else:
                    lr_policy(None, None)
            else:
                lr_policy(epoch, iteration=None)

            cur_lr = net_utils.get_lr(optimizer)
            if not cfg.no_wandb:
                wandb.log({'Epoch': epoch + (generation * end_epochs), 'LR': cur_lr})

            start_train = time.time()

            # Calculate the mask based on prune_criterion and mask_schedule (only for SNIP)
            mask_updated = False
            if cfg.prune_criterion == "SNIP" and cfg.mask_schedule != "None":
                net = deepcopy(model)
                prune = Pruner(net, dataset.train_loader, cfg.device, silent=False)
                if cfg.mask_schedule == "start_gen" and epoch == cfg.start_epoch:
                    fisher_mat = prune.snip(sparsity=cfg.sparsity)
                    mask_updated = True
                elif cfg.mask_schedule == "mid_gen" and epoch == cfg.epochs // 2:
                    fisher_mat = prune.snip(sparsity=cfg.sparsity)
                    mask_updated = True
                elif cfg.mask_schedule == "every_epoch":
                    fisher_mat = prune.snip(sparsity=cfg.sparsity)
                    mask_updated = True
                elif cfg.mask_schedule == "every_n_epochs" and (epoch + 1) % cfg.mask_n_epochs == 0:
                    fisher_mat = prune.snip(sparsity=cfg.sparsity)
                    mask_updated = True

            # Train
            if ((generation * end_epochs) + epoch) < cfg.deficit_epo and cfg.use_deficit:
                train_acc1, train_acc5, train_loss = train(
                    dataset.trainloader_deficit, model, criterion, optimizer, epoch, cfg, writer=writer, mask=fisher_mat)
            else:
                train_acc1, train_acc5, train_loss = train(
                    dataset.train_loader, model, criterion, optimizer, epoch, cfg, writer=writer, mask=fisher_mat)

            train_time.update((time.time() - start_train) / 60)

            # Store training metrics
            epoch_metrics['train_acc1'].append(float(train_acc1) if train_acc1 is not None else None)
            epoch_metrics['train_acc5'].append(float(train_acc5) if train_acc5 is not None else None)
            epoch_metrics['train_loss'].append(float(train_loss) if train_loss is not None else None)

            if not cfg.no_wandb:
                wandb.log({'Epoch': epoch + (generation * end_epochs), 'Generation': generation, 'Train Acc1': train_acc1,
                           'Train Acc5': train_acc5})

            # Validation
            if (epoch + 1) % cfg.test_interval == 0:
                start_validation = time.time()
                last_val_acc1, last_val_acc5, last_val_loss = validate(dataset.val_loader, model, criterion, cfg, writer, epoch)
                validation_time.update((time.time() - start_validation) / 60)
                epoch_metrics['test_acc1'].append(float(last_val_acc1) if last_val_acc1 is not None else None)
                epoch_metrics['test_acc5'].append(float(last_val_acc5) if last_val_acc5 is not None else None)
                epoch_metrics['test_loss'].append(float(last_val_loss) if last_val_loss is not None else None)

                if not cfg.no_wandb:
                    wandb.log(
                        {'Epoch': epoch + (generation * end_epochs), 'Generation': generation, 'Val Acc1': last_val_acc1,
                         'Val Acc5': last_val_acc5})

                is_best = last_val_acc1 > best_val_acc1
                best_val_acc1 = max(last_val_acc1, best_val_acc1)
                if last_val_acc1 * 0.999 < best_val_acc1:
                    bad_val_counter += 1
                best_val_acc5 = max(last_val_acc5, best_val_acc5)
                best_train_acc1 = max(train_acc1, best_train_acc1)
                best_train_acc5 = max(train_acc5, best_train_acc5)

                save = (((epoch + 1) % cfg.save_every) == 0) and cfg.save_every > 0

                elapsed_time = time.time() - start_time
                seconds_todo = (cfg.epochs - epoch) * (elapsed_time / cfg.test_interval)
                start_time = time.time()

                if cfg.save_model:
                    if is_best or save or epoch == cfg.epochs - 1:
                        if is_best:
                            cfg.logger.info(f"==> best {last_val_acc1:.02f} saving at {cfg.ckpt_base_dir / 'model_best.pth'}")
                        net_utils.save_checkpoint(
                            {
                                "epoch": epoch + 1,
                                "arch": cfg.arch,
                                "state_dict": model.state_dict(),
                                "best_acc1": best_val_acc1,
                                "best_acc5": best_val_acc5,
                                "best_train_acc1": best_train_acc1,
                                "best_train_acc5": best_train_acc5,
                                "best_tst_acc1": best_tst_acc1,
                                "best_tst_acc5": best_tst_acc5,
                                "optimizer": optimizer.state_dict(),
                                "curr_acc1": last_val_acc1,
                                "curr_acc5": last_val_acc5,
                            },
                            is_best,
                            filename=ckpt_base_dir / f"epoch_{epoch}.state",
                            save=save or epoch == cfg.epochs - 1,
                        )
            else:
                epoch_metrics['test_acc1'].append(None)
                epoch_metrics['test_acc5'].append(None)
                epoch_metrics['test_loss'].append(None)

            # Store mask update and sparsity
            epoch_metrics['mask_update'].append(mask_updated)
            if fisher_mat is not None:
                avg_sparsity = 0
                count = 0
                for name, param in fisher_mat.named_parameters():
                    if 'weight' in name and 'bn' not in name and 'downsample' not in name:
                        sparsity = 100 * (1 - param.mean().item())
                        epoch_metrics['layer_sparsity'][name].append(float(sparsity))
                        avg_sparsity += sparsity
                        count += 1
                avg_sparsity = avg_sparsity / count if count > 0 else 0
                epoch_metrics['avg_sparsity'].append(float(avg_sparsity))
            else:
                for name in epoch_metrics['layer_sparsity']:
                    epoch_metrics['layer_sparsity'][name].append(0.0)
                epoch_metrics['avg_sparsity'].append(0.0)

            epoch_time.update((time.time() - end_epoch) / 60)
            progress_overall.display(epoch)
            progress_overall.write_to_tensorboard(
                writer, prefix="diagnostics", global_step=epoch
            )

            writer.add_scalar("test/lr", cur_lr, epoch)
            end_epoch = time.time()

            if cfg.eval_intermediate_tst > 0 and cfg.eval_tst and (epoch + 1) % cfg.eval_intermediate_tst == 0:
                last_tst_acc1, last_tst_acc5, last_tst_loss = validate(dataset.tst_loader, model, criterion, cfg, writer, 0)
                best_tst_acc1 = max(last_tst_acc1, best_tst_acc1)
                best_tst_acc5 = max(last_tst_acc5, best_tst_acc5)
                if not cfg.no_wandb:
                    wandb.log({'Epoch': epoch, 'Test Acc1': last_tst_acc1, 'Test Acc5': last_tst_acc5})
            else:
                last_tst_acc1 = 0
                last_tst_acc5 = 0

        # Update mask at the end of generation if mask_schedule is None
        if cfg.prune_criterion in ["SNIP", "SNIPit", "SNAPit" , "CNIPit"] and cfg.mask_schedule == "None":
            net = deepcopy(model)
            prune = Pruner(net, dataset.train_loader, cfg.device, silent=False)
            if cfg.prune_criterion == "SNIP":
                fisher_mat = prune.snip(sparsity=cfg.sparsity)
            elif cfg.prune_criterion == "SNIPit":
                fisher_mat = prune.snip_it(
                    sparsity=cfg.sparsity,
                    steps=cfg.pruning_steps,
                    mini_batches=1,
                    silent=False
                )
            elif cfg.prune_criterion == "SNAPit":
                fisher_mat = prune.snap_it(
                    sparsity=cfg.sparsity,
                    steps=cfg.pruning_steps,
                    start=cfg.pruning_start,
                    mini_batches=1,
                    silent=False
                )

            elif cfg.prune_criterion == "CNIPit":
                fisher_mat = prune.cnip_it(
                    sparsity=cfg.sparsity,
                    steps=cfg.pruning_steps,
                    start=cfg.pruning_start,
                    mini_batches=1,
                    silent=False
                )    
            mask_updated = True
            epoch_metrics['mask_update'][-1] = True  # Mark the last epoch as mask-updated
            # Update sparsity metrics for the final epoch
            if fisher_mat is not None:
                avg_sparsity = 0
                count = 0
                for name, param in fisher_mat.named_parameters():
                    if 'weight' in name and 'bn' not in name and 'downsample' not in name:
                        sparsity = 100 * (1 - param.mean().item())
                        epoch_metrics['layer_sparsity'][name][-1] = float(sparsity)
                        avg_sparsity += sparsity
                        count += 1
                avg_sparsity = avg_sparsity / count if count > 0 else 0
                epoch_metrics['avg_sparsity'][-1] = float(avg_sparsity)

    except Exception as e:
        cfg.logger.error(f"Training interrupted at epoch {epoch}: {str(e)}")
        remaining_epochs = end_epochs - len(epoch_metrics['train_acc1'])
        for _ in range(remaining_epochs):
            epoch_metrics['train_acc1'].append(None)
            epoch_metrics['train_acc5'].append(None)
            epoch_metrics['train_loss'].append(None)
            epoch_metrics['test_acc1'].append(None)
            epoch_metrics['test_acc5'].append(None)
            epoch_metrics['test_loss'].append(None)
            epoch_metrics['avg_sparsity'].append(None)
            epoch_metrics['mask_update'].append(False)
            for name in epoch_metrics['layer_sparsity']:
                epoch_metrics['layer_sparsity'][name].append(0.0)

    # Handle case where end_epochs is 0
    if end_epochs == 0:
        cfg.logger.error(f"end_epochs is 0, padding metrics for {cfg.epochs} epochs")
        for _ in range(cfg.epochs):
            epoch_metrics['train_acc1'].append(None)
            epoch_metrics['train_acc5'].append(None)
            epoch_metrics['train_loss'].append(None)
            epoch_metrics['test_acc1'].append(None)
            epoch_metrics['test_acc5'].append(None)
            epoch_metrics['test_loss'].append(None)
            epoch_metrics['avg_sparsity'].append(None)
            epoch_metrics['mask_update'].append(False)
            for name in epoch_metrics['layer_sparsity']:
                epoch_metrics['layer_sparsity'][name].append(0.0)

    # Final test evaluation
    if cfg.eval_tst and cfg.eval_intermediate_tst == 0:
        last_tst_acc1, last_tst_acc5, last_tst_loss = validate(dataset.tst_loader, model, criterion, cfg, writer, 0)
        best_tst_acc1 = max(last_tst_acc1, best_tst_acc1)
        best_tst_acc5 = max(last_tst_acc5, best_tst_acc5)
        if not cfg.no_wandb:
            wandb.log({'Generation': generation, 'Test Acc1': last_tst_acc1, 'Test Acc5': last_tst_acc5})

        if len(epoch_metrics['test_acc1']) > 0:
            epoch_metrics['test_acc1'][-1] = float(last_tst_acc1) if last_tst_acc1 is not None else None
            epoch_metrics['test_acc5'][-1] = float(last_tst_acc5) if last_tst_acc5 is not None else None
            epoch_metrics['test_loss'][-1] = float(last_tst_loss) if last_tst_loss is not None else None

    # Save metrics to CSV
    col_names = ['generation', 'sparsity', 'last_val_acc1', 'last_val_acc5', 'best_val_acc1', 'best_val_acc5',
                 'last_tst_acc1', 'last_tst_acc5', 'best_tst_acc1', 'best_tst_acc5', 'best_train_acc1', 'best_train_acc5']
    arg_list = [generation, cfg.sparsity, last_val_acc1, last_val_acc5, best_val_acc1, best_val_acc5,
                last_tst_acc1, last_tst_acc5, best_tst_acc1, best_tst_acc5, best_train_acc1, best_train_acc5]

    csv_file = os.path.join(run_base_dir, "train.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, 'a') as ff:
            wr = csv.writer(ff, quoting=csv.QUOTE_ALL)
            wr.writerow(col_names)
    with open(csv_file, 'a') as ff:
        wr = csv.writer(ff, quoting=csv.QUOTE_ALL)
        wr.writerow(arg_list)

    # Save epoch metrics to CSV
    epoch_csv_file = os.path.join(run_base_dir, f"epoch_metrics_gen_{generation}.csv")
    with open(epoch_csv_file, 'w') as ff:
        writer = csv.writer(ff)
        writer.writerow(['Epoch', 'Train_Acc@1', 'Train_Acc@5', 'Train_Loss', 'Test_Acc@1', 'Test_Acc@5', 'Test_Loss', 'Avg_Sparsity', 'Mask_Update'])
        for epoch in range(cfg.epochs):
            writer.writerow([
                epoch,
                epoch_metrics['train_acc1'][epoch] if epoch < len(epoch_metrics['train_acc1']) else None,
                epoch_metrics['train_acc5'][epoch] if epoch < len(epoch_metrics['train_acc5']) else None,
                epoch_metrics['train_loss'][epoch] if epoch < len(epoch_metrics['train_loss']) else None,
                epoch_metrics['test_acc1'][epoch] if epoch < len(epoch_metrics['test_acc1']) else None,
                epoch_metrics['test_acc5'][epoch] if epoch < len(epoch_metrics['test_acc5']) else None,
                epoch_metrics['test_loss'][epoch] if epoch < len(epoch_metrics['test_loss']) else None,
                epoch_metrics['avg_sparsity'][epoch] if epoch < len(epoch_metrics['avg_sparsity']) else None,
                epoch_metrics['mask_update'][epoch] if epoch < len(epoch_metrics['mask_update']) else False
            ])

    # Save layer sparsity to CSV
    layer_sparsity_csv = os.path.join(run_base_dir, f"layer_sparsity_gen_{generation}.csv")
    with open(layer_sparsity_csv, 'w') as ff:
        writer = csv.writer(ff)
        layers = list(epoch_metrics['layer_sparsity'].keys())
        writer.writerow(['Epoch'] + layers)
        for epoch in range(cfg.epochs):
            row = [epoch]
            for layer in layers:
                row.append(epoch_metrics['layer_sparsity'][layer][epoch] if epoch < len(epoch_metrics['layer_sparsity'][layer]) else 0)
            writer.writerow(row)

    cfg.logger.info(f"==> Final Best {best_val_acc1:.02f}, saving at {cfg.ckpt_base_dir / 'model_best.pth'}")

    return run_base_dir, fisher_mat, model, epoch_metrics
    
    ################################################################################
def adjust_fisher(fim, net, filter_importance_statistic='norm'):
    layerwise_fim_norm =np.zeros(18,dtype = float)
    adjusted_fisher = fim.data
    trace_fim = (fim.trace()).detach().cpu().numpy()
    i=0
    layer_idx_dict, start_index_dict = net_utils.count_parameters(net)
    with torch.no_grad():
        for name, param in net.named_parameters():
            if ('weight' in name ) and ('downsample' not in name ) and len(param.shape) > 1:
                start_idx = start_index_dict[name]
                N = torch.numel(param.data)
                end_idx = start_idx + N
                filter_params = adjusted_fisher[start_idx: end_idx]
                if filter_importance_statistic == 'mean':
                    imp_val = torch.mean(filter_params)
                elif filter_importance_statistic == 'median':
                    imp_val = torch.median(filter_params)
                elif filter_importance_statistic == 'max':
                    imp_val = torch.max(filter_params)
                elif filter_importance_statistic == 'norm':
                    imp_val = filter_params.norm(dim=0, p=2)
                    imp_val = imp_val.detach().cpu().numpy()
                else:
                    raise ValueError('Incorrect statistic selected')
                if i <18:
                    layerwise_fim_norm[i] = imp_val
                    i = i + 1

    return layerwise_fim_norm, trace_fim



def ke_cls_eval_sparse(cfg, model, generation, ckpath, name):

    _, validate = get_trainer(cfg)
    dataset = getattr(data, cfg.set)(cfg)

    if cfg.label_smoothing == 0:
        softmax_criterion = nn.CrossEntropyLoss().cuda()
    else:
        softmax_criterion = net_utils.LabelSmoothing(smoothing=cfg.label_smoothing).cuda()

    criterion = lambda output, target: softmax_criterion(output, target)

    writer = SummaryWriter(log_dir=ckpath)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(1, [epoch_time, validation_time, train_time], cfg, prefix="Overall Timing")
    last_tst_acc1, last_tst_acc5 , tst_loss = validate(dataset.val_loader, model, criterion, cfg, writer, 0)
    if not cfg.no_wandb:
        wandb.log({'Generation': generation, 'Test Acc1': last_tst_acc1, 'Test Acc5': last_tst_acc5})
    print(name, last_tst_acc1, last_tst_acc5)

    # log results in a csv file
    col_names = ['generation', 'last_tst_acc1', 'last_tst_acc5']
    arg_list = [generation, last_tst_acc1, last_tst_acc5]
    csv_file = os.path.join(ckpath, name)
    if not os.path.exists(csv_file):
        with open(csv_file, 'a') as ff:
            wr = csv.writer(ff, quoting=csv.QUOTE_ALL)
            wr.writerow(col_names)

    with open(csv_file, 'a') as ff:
        wr = csv.writer(ff, quoting=csv.QUOTE_ALL)
        wr.writerow(arg_list )
    # torch.save(model.state_dict(), ckpath / f"sparse_model{cfg.arch}_{generation}.pth")
    return last_tst_acc1, last_tst_acc5
