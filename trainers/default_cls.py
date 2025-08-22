import time
import torch
import numpy as np
import torch.nn as nn
from utils import net_utils
from layers.CS_KD import KDLoss
from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter
#from utils.pruning import apply_reg, update_reg
import matplotlib.pyplot as plt
# Training function
import random
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms_v2
from utils import net_utils
from layers.CS_KD import KDLoss
from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter

from configs.base_config import Config

__all__ = ["train", "validate"]



kdloss = KDLoss(4).cuda()

def set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):        
        m.eval()

def set_bn_train(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.train()
                
        

__all__ = ["train", "validate"]

kdloss = KDLoss(4).cuda()

def set_bn_eval(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):        
        m.eval()

def set_bn_train(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.train()

def train(train_loader, model, criterion, optimizer, epoch, cfg, writer, mask=None):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5], cfg,
        prefix=f"Epoch: [{epoch}]",
    )

    model.train()
    
    cutmix = transforms_v2.CutMix(num_classes=cfg.num_cls)
    cutmix_prob = 0.5  

    end = time.time()

    for i, data in enumerate(train_loader):
        images, target = data[0].cuda(), data[1].long().squeeze().cuda()
        data_time.update(time.time() - end)

        if cfg.cs_kd:
            batch_size = images.size(0)
            loss_batch_size = batch_size // 2
            targets_ = target[:batch_size // 2]
            outputs = model(images[:batch_size // 2])
            loss = torch.mean(criterion(outputs, targets_))
            with torch.no_grad():
                outputs_cls = model(images[batch_size // 2:])
            cls_loss = kdloss(outputs[:batch_size // 2], outputs_cls.detach())
            lamda = 3
            loss += lamda * cls_loss
            acc1, acc5 = accuracy(outputs, targets_, topk=(1, 5))
        else:
            batch_size = images.size(0)
            loss_batch_size = batch_size
            if random.random() < cutmix_prob:
                mixed_images, mixed_target = cutmix(images, target)
                input_images = mixed_images
            else:
                input_images = images
                mixed_target = F.one_hot(target, num_classes=cfg.num_cls).float()

            output = model(input_images)
            loss = F.kl_div(F.log_softmax(output, dim=1), mixed_target, reduction='batchmean')
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), loss_batch_size)
        top1.update(acc1.item(), loss_batch_size)
        top5.update(acc5.item(), loss_batch_size)

        optimizer.zero_grad()
        loss.backward()
        if mask is not None:
            for (name, param), mask_param in zip(model.named_parameters(), mask.parameters()):
                if param.grad is not None and 'weight' in name and 'bn' not in name and 'downsample' not in name:
                    param.grad = param.grad * mask_param
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.print_freq == 0 or i == len(train_loader) - 1:
            t = (len(train_loader) * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)

    return top1.avg, top5.avg, losses.avg

#Validation function
def validate(val_loader, model, criterion, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=True)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], args, prefix="Test: "
    )


    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].cuda(), data[1].long().squeeze().cuda()
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))
        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)
        print(top1.avg, top5.avg)

    return top1.avg, top5.avg, losses.avg
