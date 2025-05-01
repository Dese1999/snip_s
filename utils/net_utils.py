import os
import math
import torch
import shutil
import models
import pathlib
import numpy as np
import torch.nn as nn
from layers import bn_type
from layers import conv_type
from layers import linear_type
import torch.backends.cudnn as cudnn
from copy import deepcopy
import matplotlib.pyplot as plt


def create_dense_mask_0(net, device, value):
    for param in net.parameters():
        param.data[param.data == param.data] = value
    net.to(device)
    return net


def get_model(args):

    args.logger.info("=> Creating model '{}'".format(args.arch))
    # args.logger.info("=> Creating model_2 '{}'".format(args.arch_2))
    
    if args.arch == 'resnet18':
        model = models.__dict__[args.arch]()
    else:
        model = models.__dict__[args.arch](args)
        if args.set =="CIFAR100" or args.set =='CIFAR10':
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()

    return model

def get_model_two(args):

    args.logger.info("=> Creating model '{}'".format(args.arch))
    # args.logger.info("=> Creating model_2 '{}'".format(args.arch_2))
    small_model = models.__dict__[args.arch](args)
    big_model = models.__dict__[args.big_arch](args)
    if args.set != "imagenet":
        small_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        big_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    if args.set == "cifar10" or args.set == "cifar100":
        small_model.maxpool = nn.Identity()
        big_model.maxpool = nn.Identity()

    return big_model, small_model

def move_model_to_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"
    # print('{}'.format(args.gpu))
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif args.multigpu is None:
        device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        args.logger.info(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )

    cudnn.benchmark = True

    return model

def save_checkpoint(state, is_best, filename="checkpoint.pth", save=False):
    filename = pathlib.Path(filename)

    if not filename.parent.exists():
        os.makedirs(filename.parent,exist_ok=True)

    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, str(filename.parent / "model_best.pth"))

        if not save:
            os.remove(filename)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def extract_slim(split_model,model):
    for (dst_n, dst_m), (src_n, src_m) in zip(split_model.named_modules(), model.named_modules()):
        if hasattr(src_m, "weight") and src_m.weight is not None:
            if hasattr(src_m, "mask"):
                src_m.extract_slim(dst_m,src_n,dst_n)
                # if src_m.__class__ == conv_type.SplitConv:
                # elif src_m.__class__ == linear_type.SplitLinear:
            elif src_m.__class__ == bn_type.SplitBatchNorm: ## BatchNorm has bn_maks not mask
                src_m.extract_slim(dst_m)




def split_reinitialize(cfg,model,reset_hypothesis=False):
    cfg.logger.info('split_reinitialize')
    # zero_reset = True
    if cfg.evolve_mode == 'zero':
        cfg.logger.info('WARNING: ZERO RESET is not optimal')
    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            if hasattr(m, "mask"): ## Conv and Linear but not BN
                assert m.split_rate < 1.0

                if reset_hypothesis and (m.__class__ == conv_type.SplitConv or  m.__class__ == linear_type.SplitLinear):
                    before_sum = torch.sum(m.mask)
                    m.reset_mask()
                    cfg.logger.info('reset_hypothesis : True {} : {} -> {}'.format(n,before_sum,torch.sum(m.mask)))
                else:
                    cfg.logger.info('reset_hypothesis : False {} : {}'.format(n, torch.sum(m.mask)))

                if m.__class__ == conv_type.SplitConv or m.__class__ == linear_type.SplitLinear:
                    m.split_reinitialize(cfg)
                else:
                    raise NotImplemented('Invalid layer {}'.format(m.__class__))


def count_parameters(model):
    total =0
    layer_idx_dict = {}
    idx = {}
    for (name, p) in model.named_parameters():
        k = p.numel()
        total+= k
        layer_idx_dict[name] = k
    keys = [k for k in layer_idx_dict.keys()]
    values = [v for v in layer_idx_dict.values()]
    for i in range(len(keys)):
        cumilative = sum(values[:i])
        idx[keys[i]] = cumilative

    return layer_idx_dict, idx   #if p.requires_grad


def extract_new_sparse_model(cfg,net, fish, generation):
    # Re-init the current task mask
    net_mask_current = create_dense_mask_0(deepcopy(net), cfg.device, value=0)
    layer_idx_dict, start_index_dict = count_parameters(net)
    # Extract sparse model for the current task
    # start_idx = 0
    dict_FIM = {}
    with torch.no_grad():
        for (name, param), param_mask in \
                zip(net.named_parameters(),
                    net_mask_current.parameters()):
            if 'weight' in name and 'bn' not in name and 'downsample' not in name:
                # if 'layer3' in name or 'layer4' in name :
                # if 'bn' not in name or not 'downsample.1' in name:
                start_idx = start_index_dict[name]
                N = torch.numel(param.data)
                end_idx = start_idx + N
                param_fish = fish[start_idx: end_idx]
                sum_fish = (param_fish).sum()
                value = sum_fish / N
                value = value.cpu().detach().numpy()
                dict_FIM[name] = value

                # param_fish = fish[start_idx: end_idx]
                param_fish = param_fish.reshape(param.shape)
                # # Exclude final linear layer weights from masking
                # if 'classifier' in name or 'linear' in name:
                #     param_mask.data[param_fish == param_fish] = 1
                #     continue

                # Select top-k higher fisher importance parameters from self.net
                if cfg.grow_sparcity_gen:
                    sparsity_gen = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
                    k = math.floor(N * sparsity_gen[generation])
                else:
                    k = math.floor(N * cfg.sparsity)
                    # print(name, N, layer_idx_dict[name], k)

                if cfg.weight_pruning:
                    sorted, indices = torch.sort(torch.flatten(param), descending=True)
                    # param_mask.data[param >= sorted[k - 1]] = 1
                    top_k_idx = indices[:k - 1]
                    param_mask = torch.flatten(param_mask)
                    param_mask.data[top_k_idx] = 1

                else:
                    sorted, indices = torch.sort(torch.flatten(param_fish), descending=True)
                    top_k_idx = indices[:k-1]
                    param_mask = torch.flatten(param_mask)
                    param_mask.data[top_k_idx] = 1
                    param_mask.reshape(param.shape)
                    # param_mask.data[param_fish > sorted[k-1]] = 1

    return net_mask_current, dict_FIM


def bar_plot(dict_values, save_path):
    # creating the dataset

    layers = ['conv1', 'layer1.0', 'layer1.0', 'layer1.1', 'layer1.1', 'layer2.0', 'layer2.0', 'layer2.1', 'layer2.1',
              'layer3.0', 'layer3.0', 'layer3.1', 'layer3.1', 'layer4.0', 'layer4.0', 'layer4.1', 'layer4.1', 'fc']

    values = list(dict_values.values())
    print(layers, values)

    fig = plt.figure(figsize=(10, 5))
    # creating the bar plot
    plt.bar(layers, values, color='maroon',
            width=0.4)
    plt.xlabel("Layers")
    plt.ylabel("sum of FIM param")
    plt.title("Sum of importance weights in each layer")
    plt.show()


def extract_non_overlapping_params(cfg,net, fish, prev_mask):
    # Re-init the current task mask
    net_mask_current = create_dense_mask_0(deepcopy(net), cfg.device, value=0)

    # Extract sparse model for the current task
    start_idx = 0
    with torch.no_grad():
        for (name, param), param_mask, param_prev_mask in \
                zip(net.named_parameters(),
                    net_mask_current.parameters(), prev_mask.parameters()):
            if 'weight' in name and 'bn' not in name and 'downsample' not in name:
                # if 'layer3' in name or 'layer4' in name :
                # if 'bn' not in name or not 'downsample.1' in name:
                N = torch.numel(param.data)
                end_idx = start_idx + N
                param_fish = fish[start_idx: end_idx]
                param_fish = param_fish.reshape(param.shape)
                param_fish[param_prev_mask == 1] = float('-inf')
                start_idx += N
                k = math.floor(N * cfg.sparsity)
                sorted, indices = torch.sort(torch.flatten(param_fish), descending=True)
                # param_mask.data[param_fish > sorted[k-1]] = 1
                top_k_idx = indices[:k - 1]
                param_mask = torch.flatten(param_mask)

                param_mask.data[top_k_idx] = 1
                param_mask.reshape(param.shape)
    return net_mask_current

def extract_sparse_weights(cfg,net, mask):
    # Re-init the current task mask
    sparse_net = deepcopy(net)

    # Extract sparse model for the current task
    with torch.no_grad():
        if cfg.snip or cfg.snipit or cfg.snapit:
            for (name, param), param_mask in \
                    zip(sparse_net.named_parameters(),
                        mask.parameters()):
                # if 'weight' in name and 'bn' not in name and 'downsample' not in name:
                if 'mask' not in name:
                    param.data = param.data * param_mask.data

        elif cfg.grasp:
            for (name, param), param_mask in \
                    zip(sparse_net.named_parameters(),
                        mask.parameters()):
                if 'weight' in name and 'bn' not in name and 'downsample' not in name:
                # if 'mask' not in name:
                    param.data = param.data * param_mask.data
    return sparse_net

    #     key = list(mask.keys())
    #     id = 0
    #     with torch.no_grad():
    #
    #         # for (name,param), (net_n,net_param) in zip(sparse_net.named_parameters(),net.named_parameters()):
    #         #     if 'mask' not in name and 'bn' not in name  and 'downsample' not in name:
    #         #         if mask[key[id]].shape == param.data.shape:
    #         #             if id in [7,12,14]:
    #         #                 continue
    #         #             param.data = param.data * mask[key[id]]
    #         #             re_init_param = re_init_weights(param.data.shape, cfg.device)
    #         #             net_param.data = net_param.data * mask[key[id]]
    #         #             re_init_param.data[mask[key[id]] == 1] = 0
    #         #             net_param.data = net_param.data + re_init_param.data
    #         #             id = id + 1
    #
    #         for idx, layer in enumerate(sparse_net.modules()):
    #             if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #                 layer.weight.data = layer.weight.data * mask[key[id]]
    #                 id = id + 1


def creating_global_mask(cfg, curr_mask, total_mask):
    # Re-init the current task mask
    # total_mask = create_dense_mask_0(deepcopy(net), cfg.device, value=0)

    # Extract sparse model for the current task
    with torch.no_grad():
        for (name, total_param_mask), param_curr_mask in \
                zip(total_mask.named_parameters(), curr_mask.parameters()):
            if 'weight' in name and 'bn' not in name and 'downsample' not in name:
                # if 'layer3' in name or 'layer4' in name :
                total_param_mask.data[param_curr_mask.data==1] = 1
                # total_param_mask.data = total_param_mask.data + param_curr_mask.data
                assert torch.max(total_param_mask.data) <= 1
    return total_mask

def reparameterize_non_sparse(cfg, net, net_sparse_set):
    # Re-initialize those params that were never part of sparse set
    for (name, param), mask_param in zip(net.named_parameters(), net_sparse_set.parameters()):
        if 'weight' in name  and 'bn' not in name and 'downsample' not in name:
           # if name not in ['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.conv1.weight' ,'layer1.1.conv2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight','layer2.1.conv1.weight','layer2.1.conv2.weight' ]:
            # if 'fc' not in name: #'layer4' not in name and
            re_init_param = re_init_weights(param.data.shape, cfg.device)
            param.data = param.data * mask_param.data
            # if name not in ['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.conv1.weight' ,'layer1.1.conv2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight','layer2.1.conv1.weight','layer2.1.conv2.weight' ]:
            #if cfg.use_shrink_perturb:
                #param.data[mask_param.data==1]= param.data[mask_param.data==1].mul_(cfg.shrink).add_(re_init_param.data[mask_param.data==1], alpha=cfg.perturb)

            re_init_param.data[mask_param.data == 1] = 0
            param.data = param.data + re_init_param.data
    return net

def renint_usnig_method(mask, method='kaiming'):
    if method == 'kaiming':
        nn.init.kaiming_uniform_(mask, a=math.sqrt(5))

    else:
        nn.init.xavier_uniform_(mask)

def structured_pruning(self, k, name, shape, param_mask, adjusted_importance, t):
    # Use k-winner take all mask first
    tmp = [int(s) for s in re.findall(r'\d+', name)]
    # FixMe: Is there a better way to handle this?
    mask_name = "self.net.layer{}[{}].kwinner{}.act_count".format(tmp[0], tmp[1], tmp[2])
    kwinner_mask = eval(mask_name)

    if self.args.non_overlaping_kwinner and t > 0:
        cumulative_mask = self.kwinner_mask[(0, mask_name)]
        for i in range(1, t):
            cumulative_mask[self.kwinner_mask[(i, mask_name)] == 1] = 1
        kwinner_mask[cumulative_mask == 1] = 0
    _, indices_1 = torch.topk(kwinner_mask, k)
    mask = torch.empty(shape[0], device=self.device).fill_(float('-inf'))
    mask.scatter_(0, indices_1, 1)

    # Log the mask
    self.kwinner_mask[(t, mask_name)] = mask

    # Prune the layer based on k-winner mask
    num_filters = shape[0]
    if 'conv' in name and len(shape) > 1:
        for filter_idx in range(num_filters):
            if mask[filter_idx] < 0:
                adjusted_importance[filter_idx, :, :, :] = mask[filter_idx]

        N = (mask == 1).sum() * torch.numel(adjusted_importance[0, :, :, :])
        l = math.floor(N * self.sparsity)
        indices_2 = torch.topk(torch.flatten(adjusted_importance), l)[1]
        pruning_mask = torch.zeros(torch.numel(adjusted_importance), device=self.device)
        pruning_mask.scatter_(0, indices_2, 1)
        param_mask.data += pruning_mask.reshape(shape)
    else:
        # for filter_idx in range(num_filters):
        #     if mask[filter_idx] < 0:
        #         adjusted_importance[filter_idx] = mask[filter_idx]
        adjusted_importance[mask < 0] = float('-inf')
        N = (mask == 1).sum()
        l = math.floor(N * self.sparsity)
        # l = math.floor(shape[0] * self.sparsity)
        indices_2 = torch.topk(torch.flatten(adjusted_importance), l)[1]
        param_mask.data[indices_2] = 1


def re_init_weights(shape, device, reinint_method='kaiming'):
    mask = torch.empty(shape, requires_grad=False, device=device)
    if len(mask.shape) < 2:
        mask = torch.unsqueeze(mask, 1)
        renint_usnig_method(mask, reinint_method)
        mask = torch.squeeze(mask, 1)
    else:
        renint_usnig_method(mask, reinint_method)
    return mask

def diff_lr_sparse_dense(self):
    # LR multiplier for weights in sparse set, rest remain the same
    for param_net, param_sparse in zip(self.net.parameters(), self.net_sparse_set.parameters()):
        param_lr = torch.ones(param_sparse.data.shape, device=self.device)
        param_lr[param_sparse == 1] = self.slow_lr_multiplier
        param_net.grad = param_net.grad * param_lr

def diff_lr_sparse(cfg, net, net_sparse_set):
    # Update gradients of current sparse mask based on whether they are in sparse set
    for (name, param_net),  param_sparse in zip(net.named_parameters(),
                                                      net_sparse_set.parameters()):

        if 'weight' in name and 'bn' not in name and 'downsample' not in name:
            param_lr = torch.ones(param_sparse.data.shape, device=cfg.device)
                # Slow learning rate for overlapping weights' gradients
            param_lr[param_sparse==1] = cfg.slow_lr_multiplier
            # No gradient for weights that are not part of current sparse mask
            # param_lr[param_current == 0] = 0
            param_net.grad = param_net.grad * param_lr
    return net



def split_reinitialize_proj(cfg, model, reset_hypothesis=False):
    cfg.logger.info('split_reinitialize proj')
    # zero_reset = True
    if cfg.evolve_mode == 'zero':
        cfg.logger.info('WARNING: ZERO RESET is not optimal')
    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            if isinstance(m, nn.Linear):
                # rand_tensor = torch.zeros_like(m.weight).cuda()
                nn.init.kaiming_normal_(m.weight)
                # m.weight.data = torch.where(m.weight.data, rand_tensor)
                # torch.nn.init.xavier_uniform(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    # if cfg.evolve_mode == 'rand':
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
                    # m.bias.data.fill_(0.01)


    if cfg.reset_bn:
        for n, m in model.named_modules():
            if 'bn' in n or 'downsample.1' in n:
                m.weight.data = torch.ones(m.weight.data.shape)
                m.bias.data = torch.zeros(m.bias.data.shape)
                m.running_mean = torch.zeros(m.running_mean.shape)
                m.running_var = torch.zeros(m.running_var.shape)
                m.num_batches_tracked = torch.tensor(0)





class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def reset_mask(cfg,model):
    cfg.logger.info("=> reseting model mask")

    for n, m in model.named_modules():
        if hasattr(m, "mask"):
            cfg.logger.info(f"==> reset {n}.mask")
            # m.mask.requires_grad = True
            m.reset_mask()

        if hasattr(m, "bias_mask"):
            cfg.logger.info(f"==> reset {n}.bias_mask")
            m.reset_bias_mask()
            # m.bias_mask.requires_grad = True

def load_pretrained(pretrained_path,gpus, model,cfg):
    if os.path.isfile(pretrained_path):
        cfg.logger.info("=> loading pretrained weights from '{}'".format(pretrained_path))
        pretrained = torch.load(
            pretrained_path,
            map_location=torch.device("cuda:{}".format(gpus)),
        )["state_dict"]
        
        skip = ' '
        
        model_state_dict = model.state_dict()
        for k, v in pretrained.items():
            # if k not in model_state_dict or v.size() != model_state_dict[k].size():
            if k not in model_state_dict or v.size() != model_state_dict[k].size() or skip in k:
                cfg.logger.info("IGNORE: {}".format(k))
        pretrained = {
            k: v
            for k, v in pretrained.items()
            if (k in model_state_dict and v.size() == model_state_dict[k].size() and skip not in k)
        }
        model_state_dict.update(pretrained)
        model.load_state_dict(model_state_dict)

    else:
        cfg.logger.info("=> no pretrained weights found at '{}'".format(pretrained_path))
        raise Exception("=> no pretrained weights found at '{}'".format(pretrained_path))




