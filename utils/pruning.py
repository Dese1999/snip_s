import matplotlib.pyplot as plt
from torchvision import *
import numpy as np
import torch
from copy import deepcopy
import utils
from utils.net_utils import create_dense_mask_0
import torch.nn as nn
import torch.autograd as autograd

class Pruner:
    def __init__(self, model, loader=None, device='cpu', silent=False):
        self.device = device
        self.loader = loader
        self.model = model
        
        self.weights = [layer for name, layer in model.named_parameters() if 'mask' not in name]
        self.indicators = [torch.ones_like(layer) for name, layer in model.named_parameters() if 'mask' not in name]
        self.mask_ = utils.net_utils.create_dense_mask_0(deepcopy(model), self.device, value=1)
        self.pruned = [0 for _ in range(len(self.indicators))]
 
        if not silent:
            print("number of weights to prune:", [x.numel() for x in self.indicators])

    def indicate(self):
        """
        Apply indicators (masks) to the model weights.
        """
        with torch.no_grad():
            for weight, indicator in zip(self.weights, self.indicators):
                weight.data = weight.data * indicator.to(weight.device)
            # Update mask_ to reflect indicators
            idx = 0
            for name, param in self.mask_.named_parameters():
                if 'mask' not in name:
                    param.data = self.indicators[idx].data
                    idx += 1

    def snip(self, sparsity, mini_batches=1, silent=False):  # prunes due to SNIP method
        mini_batches = len(self.loader) / 32
        mini_batch = 0
        self.indicate()
        self.model.zero_grad()
        grads = [torch.zeros_like(w) for w in self.weights]
        
        for x, y in self.loader:
            x, y = x.to(self.device), y.to(self.device)
            x = self.model.forward(x)
            L = torch.nn.CrossEntropyLoss()(x, y)
            grads = [g.abs() + (ag.abs() if ag is not None else torch.zeros_like(g)) 
                     for g, ag in zip(grads, torch.autograd.grad(L, self.weights, allow_unused=True))]
            
            mini_batch += 1
            if mini_batch >= mini_batches: 
                break

        with torch.no_grad():
            saliences = [(grad * weight).view(-1).abs().cpu() for weight, grad in zip(self.weights, grads)]
            saliences = torch.cat(saliences)
            
            thresh = float(saliences.kthvalue(int(sparsity * saliences.shape[0]))[0])
            
            for j, layer in enumerate(self.indicators):
                layer[(grads[j] * self.weights[j]).abs() <= thresh] = 0
                self.pruned[j] = int(torch.sum(layer == 0))
        
        self.indicate()
        
        current_sparsity = sum(self.pruned) / sum([ind.numel() for ind in self.indicators])
        
        if not silent:
            print("weights left: ", [self.indicators[i].numel() - pruned for i, pruned in enumerate(self.pruned)])
            print("sparsities: ", [round(100 * pruned / self.indicators[i].numel(), 2) for i, pruned in enumerate(self.pruned)])
            print(f"Current total sparsity: {current_sparsity*100:.2f}\n")
        
        return self.mask_
        
    def lottery_ticket_prune(self, sparsity, steps=5, reinit=True, silent=False):
        """
        Apply Lottery Ticket Hypothesis (LTH) pruning with layer-wise iterative magnitude pruning.
        Args:
            sparsity (float): Target sparsity level (0 to 1).
            steps (int): Number of iterative pruning steps.
            reinit (bool): Whether to reinitialize weights to their original values after pruning.
            silent (bool): If True, suppress printing.
        Returns:
            Updated mask_ (torch.nn.Module).
        """
        if not silent:
            print(f"LTH: Targeting sparsity {sparsity:.4f} over {steps} steps")

        # Store original weights for reinitialization
        original_params = [p.clone().detach() for p in self.weights]

        # Calculate per-step sparsity targets
        start_sparsity = 0.0
        prune_steps = [start_sparsity + (sparsity - start_sparsity) * (1 - (1 - (i + 1) / steps) ** 2) for i in range(steps)]
        prune_steps[-1] = sparsity  # Ensure final step reaches target sparsity

        for step, target_sparsity in enumerate(prune_steps):
            if not silent:
                print(f"LTH Step {step + 1}/{steps}: Targeting sparsity {target_sparsity:.4f}")

            with torch.no_grad():
                for j, (weight, layer) in enumerate(zip(self.weights, self.indicators)):
                    # Consider only active weights (where indicator == 1)
                    active_weights = weight[layer == 1].view(-1).abs().cpu()
                    if active_weights.numel() == 0:
                        continue  # Skip if no active weights remain
                    # Calculate layer-specific threshold
                    cutoff_index = int(round(target_sparsity * active_weights.numel()))
                    cutoff_index = min(cutoff_index, active_weights.numel() - 1)  # Avoid out-of-bounds
                    sorted_weights, _ = torch.sort(active_weights)
                    thresh = float(sorted_weights[cutoff_index])
                    # Apply pruning to the layer
                    layer[(weight.abs() <= thresh) & (layer == 1)] = 0
                    self.pruned[j] = int(torch.sum(layer == 0))

            self.indicate()

            # Calculate current total sparsity
            current_sparsity = sum(self.pruned) / sum([ind.numel() for ind in self.indicators])
            if not silent:
                print("Weights left: ", [self.indicators[i].numel() - pruned for i, pruned in enumerate(self.pruned)])
                print("Sparsities: ", [round(100 * p / i.numel(), 2) for i, p in zip(self.indicators, self.pruned)])
                print(f"Current total sparsity: {current_sparsity*100:.2f}\n")

            if reinit:
                with torch.no_grad():
                    for p, orig_p in zip(self.weights, original_params):
                        p.copy_(orig_p)

            if abs(current_sparsity - sparsity) < 1e-3:
                break

        return self.mask_    

    def snip_it(self, sparsity, steps=5, mini_batches=1, silent=False):
        start = 0.5
        prune_steps = [sparsity - (sparsity - start) * (0.5 ** i) for i in range(steps)] + [sparsity]
        print(f"prune_steps: {prune_steps}")
        for step, target_sparsity in enumerate(prune_steps):
            if not silent:
                print(f"SNIP-it step {step + 1}/{len(prune_steps)}: Targeting sparsity {target_sparsity:.4f}")
            
            self.indicate()
            self.model.zero_grad()
            grads = [torch.zeros_like(w) for w in self.weights]
            loss = 0.0
            mini_batch = 0
            
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                x = self.model.forward(x)
                L = torch.nn.CrossEntropyLoss()(x, y)
                loss += L.item()
                grads = [g.abs() + (ag.abs() if ag is not None else torch.zeros_like(g)) 
                         for g, ag in zip(grads, torch.autograd.grad(L, self.weights, allow_unused=True))]
                
                mini_batch += 1
                if mini_batch >= mini_batches: 
                    break
            
            loss /= max(1, mini_batch)
            
            with torch.no_grad():
                saliences = [(grad * weight).view(-1).abs() / (loss + 1e-8) 
                             for grad, weight in zip(grads, self.weights)]
                saliences = torch.cat(saliences).cpu()
                thresh = float(saliences.kthvalue(int(target_sparsity * saliences.shape[0]))[0])
                
                for j, (grad, weight, layer) in enumerate(zip(grads, self.weights, self.indicators)):
                    layer[(grad * weight).abs() / (loss + 1e-8) <= thresh] = 0
                    self.pruned[j] = int(torch.sum(layer == 0))
            
            self.indicate()
            
            current_sparsity = sum(self.pruned) / sum([ind.numel() for ind in self.indicators])
            
            if not silent:
                print("Step weights left: ", [self.indicators[i].numel() - pruned for i, pruned in enumerate(self.pruned)])
                print("Step sparsities: ", [round(100 * pruned / self.indicators[i].numel(), 2) 
                                          for i, pruned in enumerate(self.pruned)])
                print(f"Current total sparsity: {current_sparsity*100:.2f}\n")
            
            if abs(current_sparsity - sparsity) < 1e-3:
                break
        
        return self.mask_

    def snap_it(self, sparsity, steps=5, start=0.5, mini_batches=1, silent=False):
        prune_steps = [sparsity - (sparsity - start) * (0.5 ** i) for i in range(steps)] + [sparsity]
        current_sparsity = 0.0
        remaining = 1.0
        
        for step, target_sparsity in enumerate(prune_steps):
            if not silent:
                print(f"SNAP-it step {step + 1}/{len(prune_steps)}: Targeting sparsity {target_sparsity:.4f}")
            
            prune_rate = (target_sparsity - current_sparsity) / (remaining + 1e-8)
            
            self.indicate()
            self.model.zero_grad()
            grads = [torch.zeros_like(w) for w in self.weights]
            loss = 0.0
            mini_batch = 0
            
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                x = self.model.forward(x)
                L = torch.nn.CrossEntropyLoss()(x, y)
                loss += L.item()
                grads = [g.abs() + (ag.abs() if ag is not None else torch.zeros_like(g)) 
                         for g, ag in zip(grads, torch.autograd.grad(L, self.weights, allow_unused=True))]
                
                mini_batch += 1
                if mini_batch >= mini_batches: 
                    break
            
            loss /= max(1, mini_batch)
            
            with torch.no_grad():
                saliences = []
                layer_names = [name for name, _ in self.model.named_parameters() if 'mask' not in name]
                for j, (grad, weight, layer) in enumerate(zip(grads, self.weights, self.indicators)):
                    if len(weight.shape) == 4:  # Conv2d layer
                        importance = torch.sum(grad.abs(), dim=(1, 2, 3)) / (loss + 1e-8)
                        saliences.append(importance.view(-1))
                    else:
                        importance = (grad * weight).abs().view(-1) / (loss + 1e-8)
                        saliences.append(importance)
                
                saliences = torch.cat(saliences).cpu()
                thresh = float(saliences.kthvalue(int(prune_rate * saliences.shape[0]))[0])
                
                for j, (grad, weight, layer) in enumerate(zip(grads, self.weights, self.indicators)):
                    if len(weight.shape) == 4:  # Conv2d layer
                        importance = torch.sum((grad * weight).abs(), dim=(1, 2, 3)) / (loss + 1e-8)
                        layer[importance <= thresh, :, :, :] = 0
                    else:
                        layer[(grad * weight).abs() / (loss + 1e-8) <= thresh] = 0
                    self.pruned[j] = int(torch.sum(layer == 0))
            
            self.indicate()
            
            current_sparsity = sum(self.pruned) / sum([ind.numel() for ind in self.indicators])
            remaining = 1.0 - current_sparsity
            
            if not silent:
                print("Step weights left: ", [self.indicators[i].numel() - pruned for i, pruned in enumerate(self.pruned)])
                print("Step sparsities: ", [round(100 * pruned / self.indicators[i].numel(), 2) 
                                          for i, pruned in enumerate(self.pruned)])
                print(f"Current total sparsity: {current_sparsity*100:.2f}\n")
            
            if abs(current_sparsity - sparsity) < 1e-3:
                break
        
        return self.mask_

    def cnip_it(self, sparsity, steps=5, start=0.5, mini_batches=1, silent=False):
            """
            Apply CNIP-it pruning (iterative, combined unstructured and structured) based on weight and node elasticity.
            Args:
                sparsity (float): Target sparsity level (0 to 1).
                steps (int): Number of pruning steps.
                start (float): Starting sparsity for the pruning schedule.
                mini_batches (int): Number of mini-batches to compute gradients.
                silent (bool): If True, suppress printing.
            Returns:
                Updated mask_ (torch.nn.Module): The pruned mask.
            """
            # Calculate pruning steps
            prune_steps = [sparsity - (sparsity - start) * (0.5 ** i) for i in range(steps)] + [sparsity]
            current_sparsity = 0.0
            remaining = 1.0

            for step, target_sparsity in enumerate(prune_steps):
                if not silent:
                    print(f"CNIP-it step {step + 1}/{len(prune_steps)}: Targeting sparsity {target_sparsity:.4f}")

                # Calculate pruning rate for this step
                prune_rate = (target_sparsity - current_sparsity) / (remaining + 1e-8)

                # Compute gradients
                self.indicate()  # Ensure indicators are updated
                self.model.zero_grad()
                grads = [torch.zeros_like(w) for w in self.weights]
                loss = 0.0
                mini_batch = 0

                for x, y in self.loader:
                    x, y = x.to(self.device), y.to(self.device)
                    output = self.model.forward(x)
                    L = nn.CrossEntropyLoss()(x, y)
                    loss += L.item()
                    grad_outputs = torch.autograd.grad(L, self.weights, allow_unused=True)
                    grads = [g.abs() + (ag.abs() if ag is not None else torch.zeros_like(g))
                            for g, ag in zip(grads, grad_outputs)]
                    mini_batch += 1
                    if mini_batch >= mini_batches:
                        break

                if mini_batch == 0:
                    raise ValueError("No mini-batches processed. Check data loader.")
                loss /= mini_batch

                with torch.no_grad():
                    # Compute weight and node saliencies
                    weight_saliences = []
                    node_saliences = []
                    for j, (grad, weight) in enumerate(zip(grads, self.weights)):
                        # Weight-elasticity (SNIP-like)
                        weight_importance = (grad * weight).abs().view(-1) / (loss + 1e-8)
                        weight_saliences.append(weight_importance)

                        # Node-elasticity for Conv2d layers (SNAP-like)
                        if len(weight.shape) == 4:  # Conv2d layer
                            node_importance = torch.sum(grad.abs(), dim=(1, 2, 3)) / (loss + 1e-8)
                            if not silent:
                                print(f"Layer {j}: Weight shape: {weight.shape}, Node importance shape: {node_importance.shape}")
                            node_saliences.append(node_importance.view(-1))
                        else:
                            node_saliences.append(torch.zeros_like(weight_importance))

                    # Combine saliencies and determine threshold
                    all_saliences = torch.cat(weight_saliences + node_saliences).cpu()
                    if all_saliences.numel() == 0:
                        raise ValueError("No saliencies computed. Check model weights or gradients.")
                    thresh = float(all_saliences.kthvalue(int(prune_rate * all_saliences.shape[0]))[0])

                    # Determine pruning thresholds
                    weight_threshold = thresh
                    node_threshold = thresh
                    percentage_weights = sum((ws < weight_threshold).sum().item() for ws in weight_saliences) / sum(ws.numel() for ws in weight_saliences) if weight_saliences else 0.0
                    percentage_nodes = sum((ns < node_threshold).sum().item() for ns in node_saliences) / sum(ns.numel() for ns in node_saliences) if node_saliences else 0.0

                    if not silent:
                        print(f"Fraction for pruning nodes: {percentage_nodes:.4f}, Fraction for pruning weights: {percentage_weights:.4f}")

                    # Prune weights and nodes separately
                    for j, (grad, weight, layer) in enumerate(zip(grads, self.weights, self.indicators)):
                        # Prune weights (SNIP-like)
                        weight_mask = (grad * weight).abs() / (loss + 1e-8) >= weight_threshold
                        layer[weight_mask == False] = 0

                        # Prune nodes for Conv2d layers (SNAP-like)
                        if len(weight.shape) == 4:  # Conv2d layer
                            node_importance = torch.sum(grad.abs(), dim=(1, 2, 3)) / (loss + 1e-8)
                            node_mask = node_importance >= node_threshold
                            if node_mask.shape[0] != weight.shape[0]:
                                raise ValueError(f"Node mask dimension mismatch: {node_mask.shape} vs expected [{weight.shape[0]}]")
                            layer[node_mask == False, :, :, :] = 0  # Zero out entire channels

                        self.pruned[j] = int(torch.sum(layer == 0))

                self.indicate()  # Update indicators after pruning

                # Update sparsity metrics
                total_params = sum(ind.numel() for ind in self.indicators)
                current_sparsity = sum(self.pruned) / total_params if total_params > 0 else 0.0
                remaining = 1.0 - current_sparsity

                if not silent:
                    print("Step weights left: ", [ind.numel() - pruned for ind, pruned in zip(self.indicators, self.pruned)])
                    print("Step sparsities: ", [round(100 * pruned / ind.numel(), 2)
                                            for ind, pruned in zip(self.indicators, self.pruned)])
                    print(f"Current total sparsity: {current_sparsity*100:.2f}\n")

                if abs(current_sparsity - sparsity) < 1e-3:
                    break

            if self.mask_ is None:
                raise ValueError("Mask not generated. Check pruning process.")
            return self.mask_

    def snipR(self, sparsity, silent=False):
        with torch.no_grad():
            saliences = [torch.zeros_like(w) for w in self.weights]
            x, y = next(iter(self.loader))
            x, y = x.to(self.device), y.to(self.device)
            z = self.model.forward(x)
            L0 = torch.nn.CrossEntropyLoss()(z, y)

            for laynum, layer in enumerate(self.weights):
                if not silent:
                    print("layer ", laynum, "...")
                for weight in range(layer.numel()):
                    temp = layer.view(-1)[weight].clone()
                    layer.view(-1)[weight] = 0
                    z = self.model.forward(x)
                    L = torch.nn.CrossEntropyLoss()(z, y)
                    saliences[laynum].view(-1)[weight] = (L - L0).abs()    
                    layer.view(-1)[weight] = temp
                
            saliences_bag = torch.cat([s.view(-1) for s in saliences]).cpu()
            thresh = float(saliences_bag.kthvalue(int(sparsity * saliences_bag.numel()))[0])

            for j, layer in enumerate(self.indicators):
                layer[saliences[j] <= thresh] = 0
                self.pruned[j] = int(torch.sum(layer == 0))   
        
        self.indicate()
        
        if not silent:
            print("weights left: ", [self.indicators[i].numel() - pruned for i, pruned in enumerate(self.pruned)])
            print("sparsities: ", [round(100 * pruned / self.indicators[i].numel(), 2) 
                                  for i, pruned in enumerate(self.pruned)])

    def cwi_importance(self, sparsity, device):
        mask = utils.net_utils.create_dense_mask_0(deepcopy(self.model), device, value=0)
        for (name, param), param_mask in zip(self.model.named_parameters(), mask.parameters()):
            if 'weight' in name and 'bn' not in name and 'downsample' not in name:
                param_mask.data = abs(param.data) + abs(param.grad)

        imp = [layer for name, layer in mask.named_parameters() if 'mask' not in name]
        imp = torch.cat([i.view(-1).cpu() for i in imp])
        percentile = np.percentile(imp.numpy(), sparsity * 100)
        above_threshold = [i > percentile for i in imp]
        for i, param_mask in enumerate(mask.parameters()):
            param_mask.data = param_mask.data * above_threshold[i].view(param_mask.shape).to(device)
        return mask

    def apply_reg(self, mask):
        for (name, param), param_mask in zip(self.model.named_parameters(), mask.parameters()):
            if 'weight' in name and 'bn' not in name and 'downsample' not in name:
                l2_grad = param_mask.data * param.data
                param.grad += l2_grad

    def update_reg(self, mask, reg_decay, cfg):
        reg_mask = create_dense_mask_0(deepcopy(mask), cfg.device, value=0)
        for (name, param), param_mask in zip(reg_mask.named_parameters(), mask.parameters()):
            if 'weight' in name and 'bn' not in name and 'downsample' not in name:
                param.data[param_mask.data == 1] = 0
                if cfg.reg_type == 'x':
                    if reg_decay < 1:
                        param.data[param_mask.data == 0] += min(reg_decay, 1)
                elif cfg.reg_type == 'x^2':
                    if reg_decay < 1:
                        param.data[param_mask.data == 0] += min(reg_decay, 1)
                        param.data[param_mask.data == 0] = param.data[param_mask.data == 0] ** 2
                elif cfg.reg_type == 'x^3':
                    if reg_decay < 1:
                        param.data[param_mask.data == 0] += min(reg_decay, 1)
                        param.data[param_mask.data == 0] = param.data[param_mask.data == 0] ** 3
        reg_decay += cfg.reg_granularity_prune
        return reg_mask, reg_decay

    def grasp(self, sparsity, num_classes, samples_per_class=25, num_iters=1, T=200, reinit=True, silent=False):
        """
        Apply GraSP pruning based on Hessian-based importance scores.
        Args:
            sparsity (float): Target sparsity level (0 to 1).
            num_classes (int): Number of classes in the dataset.
            samples_per_class (int): Number of samples per class to fetch.
            num_iters (int): Number of iterations for gradient computation.
            T (float): Temperature parameter for softmax.
            reinit (bool): Whether to reinitialize weights after pruning.
            silent (bool): If True, suppress printing.
        Returns:
            Updated mask_ (torch.nn.Module).
        """
        if not silent:
            print(f"GraSP: Targeting sparsity {sparsity:.4f}")

        # Fetch data
        inputs, targets = self.fetch_data(self.loader, num_classes, samples_per_class)
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        scores = torch.zeros(sum(p.numel() for p in self.weights)).to(self.device)
        original_params = [p.clone().detach() for p in self.weights]

        for it in range(num_iters):
            self.model.zero_grad()
            outputs = self.model(inputs)
            softmax_outputs = nn.functional.softmax(outputs / T, dim=1)

            for c in range(num_classes):
                class_mask = (targets == c).float()
                if class_mask.sum() == 0:
                    continue
                class_outputs = softmax_outputs[:, c]
                class_loss = -(class_mask * torch.log(class_outputs + 1e-10)).sum() / class_mask.sum()
                class_loss.backward(retain_graph=True)

                # Compute gradient norm
                grad_norm = 0
                for p in self.weights:
                    if p.grad is not None:
                        grad_norm += torch.norm(p.grad) ** 2
                grad_norm = torch.sqrt(grad_norm + 1e-10)

                # Accumulate scores
                idx = 0
                for p in self.weights:
                    if p.grad is not None:
                        num_params = p.numel()
                        scores[idx:idx + num_params] += (p.grad.view(-1) * p.view(-1)) / grad_norm
                        idx += num_params

                self.model.zero_grad()

        # Normalize scores
        scores = scores / (num_iters * num_classes)
        scores = scores.cpu()

        # Compute threshold for pruning
        thresh = float(torch.kthvalue(scores.abs(), int(sparsity * scores.shape[0]))[0])

        # Apply pruning
        with torch.no_grad():
            idx = 0
            for j, layer in enumerate(self.indicators):
                num_params = self.weights[j].numel()
                layer_mask = (scores[idx:idx + num_params].abs() <= thresh).view(layer.shape)
                layer[layer_mask] = 0
                self.pruned[j] = int(torch.sum(layer == 0))
                idx += num_params

        self.indicate()

        # Calculate current sparsity
        current_sparsity = sum(self.pruned) / sum([ind.numel() for ind in self.indicators])

        if not silent:
            print("Weights left: ", [self.indicators[i].numel() - pruned for i, pruned in enumerate(self.pruned)])
            print("Sparsities: ", [round(100 * pruned / self.indicators[i].numel(), 2) for i, pruned in enumerate(self.pruned)])
            print(f"Current total sparsity: {current_sparsity*100:.2f}\n")

        if reinit:
            with torch.no_grad():
                for p, orig_p in zip(self.weights, original_params):
                    p.copy_(orig_p)

        return self.mask_

    def fetch_data(self, dataloader, num_classes, samples_per_class):
        """
        Fetch a balanced subset of data for GraSP.
        """
        class_counts = [0] * num_classes
        inputs = []
        targets = []
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            for i in range(y.size(0)):
                label = y[i].item()
                if class_counts[label] < samples_per_class:
                    inputs.append(x[i:i+1])
                    targets.append(y[i:i+1])
                    class_counts[label] += 1
                if all(count >= samples_per_class for count in class_counts):
                    break
            if all(count >= samples_per_class for count in class_counts):
                break
        inputs = torch.cat(inputs, dim=0)
        targets = torch.cat(targets, dim=0)
        return inputs, targets

    def lottery_ticket_prune(self, sparsity, steps=5, reinit=True, silent=False, use_attention=True, alpha=0.7, attention_type='se'):
        if not silent:
            print(f"AttnLTH: Targeting sparsity {sparsity:.4f} over {steps} steps")
            if use_attention:
                print(f"Using {attention_type} attention with alpha={alpha:.2f}")
            print(f"Dataloader: {self.dataloader}")
    
        original_params = [p.clone().detach() for p in self.weights]
        start_sparsity = 0.0
        prune_steps = [start_sparsity + (sparsity - start_sparsity) * (1 - (1 - (i + 1) / steps) ** 2) 
                       for i in range(steps)]
        prune_steps[-1] = sparsity
    
        for step, target_sparsity in enumerate(prune_steps):
            if not silent:
                print(f"AttnLTH Step {step + 1}/{steps}: Targeting sparsity {target_sparsity:.4f}")
    
            with torch.no_grad():
                attention_weights = {}
                if use_attention and attention_type == 'se':
                    se_count = 0
                    if not silent:
                        print("Scanning model for SE blocks...")
                    for name, module in self.model.named_modules():
                        if not silent:
                            print(f"Checking module: {name}")
                        if hasattr(module, 'se'):
                            if not silent:
                                print(f"Found 'se' attribute in module {name}")
                            if isinstance(module.se, SEBlock):
                                try:
                                    # محاسبه وزن‌های توجه
                                    attention = module.se.compute_average_attention(self.dataloader)
                                    # اصلاح نام ماژول برای مطابقت با لایه‌های کانو
                                    conv_module_name = f"{name}.conv2" if 'layer' in name else name
                                    attention_weights[conv_module_name] = attention
                                    se_count += 1
                                    if not silent:
                                        print(f"SE block found in module {name}, stored as {conv_module_name}, attention weights shape: {attention.shape}")
                                except Exception as e:
                                    if not silent:
                                        print(f"Error computing attention for module {name}: {str(e)}")
                    if not silent:
                        print(f"Total SE blocks detected: {se_count}")
                        if se_count == 0:
                            print("Warning: No SE blocks found in the model!")
                        else:
                            print(f"Attention weights keys: {list(attention_weights.keys())}")
    
                for j, (weight, layer) in enumerate(zip(self.weights, self.indicators)):
                    importance = weight.abs().mean(dim=[1,2,3] if len(weight.shape) == 4 else 1)
                    module_name = None
                    for name, param in self.model.named_parameters():
                        if param is weight:
                            module_name = name.replace('.weight', '')
                            break
                    if not silent:
                        print(f"Processing layer: {module_name}, importance shape: {importance.shape}")
    
                    if use_attention and attention_type == 'se':
                        if module_name in attention_weights:
                            attn = attention_weights[module_name]
                            if attn.shape == importance.shape:
                                importance = alpha * importance + (1 - alpha) * attn
                                if not silent:
                                    print(f"Applied attention for layer {module_name}, importance shape: {importance.shape}")
                            else:
                                if not silent:
                                    print(f"Warning: Attention weights shape {attn.shape} does not match importance shape {importance.shape} for layer {module_name}")
                        else:
                            if not silent:
                                print(f"Warning: No attention weights found for layer {module_name}")
    
                    active_weights = importance[layer == 1].view(-1).cpu()
                    if active_weights.numel() == 0:
                        if not silent:
                            print(f"Skipping layer {module_name}: MacBook Air M3, MacBook Pro M3, MacBook Pro 16-Inch, MacBook Air M2, MacBook Pro 14-Inch, MacBook Air 13-Inch, MacBook Pro 13-Inch, MacBook Air M1, MacBook Pro, MacBook Air 11-Inch, MacBook, MacBook Air, MacBook 12-Inch, MacBook Pro 15-Inch, MacBook 15-Inch, MacBook 13-Inch Retina, MacBook Pro 13-Inch Retina, MacBook Air 13-Inch Retina, MacBook Air 13, MacBook Pro 13, MacBook 13, MacBook Pro 15, MacBook 15, MacBook Air 11, MacBook Pro 17-Inch, MacBook 17-Inch, MacBook Pro 13-Inch Touch Bar, MacBook Pro 15-Inch Touch Bar, MacBook Air 13-Inch Early 2015, MacBook Pro 13-Inch Early 2015, MacBook Air 11-Inch Early 2015, MacBook Pro 15-Inch Early 2013, MacBook Pro 13-Inch Late 2013, MacBook Air 13-Inch Late 2013, MacBook Pro 15-Inch Late 2013, MacBook Air 11-Inch Mid 2013, MacBook Pro 13-Inch Mid 2012, MacBook Air 11-Inch Mid 2012, MacBook Pro 15-Inch Mid 2012, MacBook 17-Inch Late 2011, MacBook Air 11-Inch Late 2010, MacBook Pro 15-Inch Early 2011, MacBook Pro 13-Inch Early 2011, MacBook Air 13-Inch Late 2010, MacBook Pro 15-Inch Mid 2010, MacBook Pro 13-Inch Mid 2010, MacBook Air 11-Inch Late 2010, MacBook Pro 17-Inch Mid 2010, MacBook Pro 15-Inch Mid 2009, MacBook Pro 17-Inch Mid 2009, MacBook Air Mid 2009, MacBook Pro 15-Inch Late 2008, MacBook Air Late 2008, MacBook Pro 15-Inch Early 2008, MacBook Air Early 2008, MacBook Pro 15-Inch Late 2007, MacBook Pro 17-Inch Late 2007, MacBook Pro 15-Inch Mid 2007, MacBook Pro 17-Inch Mid 2007, MacBook Pro 15-Inch Late 2006, MacBook Pro 17-Inch Late 2006, MacBook Pro 15-Inch 2006, MacBook Pro 17-Inch 2006 due to no active weights")
                        continue
                    cutoff_index = int(round(target_sparsity * active_weights.numel()))
                    cutoff_index = min(cutoff_index, active_weights.numel() - 1)
                    sorted_weights, _ = torch.sort(active_weights)
                    thresh = float(sorted_weights[cutoff_index])
                    prune_mask = (importance <= thresh) & (layer == 1)
                    layer[prune_mask] = 0
                    self.pruned[j] = int(torch.sum(layer == 0))
    
            self.indicate()
            current_sparsity = sum(self.pruned) / sum([ind.numel() for ind in self.indicators])
            if not silent:
                print("Weights left: ", [self.indicators[i].numel() - pruned 
                                      for i, pruned in enumerate(self.pruned)])
                print("Sparsities: ", [round(100 * p / i.numel(), 2) 
                                      for i, p in zip(self.indicators, self.pruned)])
                print(f"Current total sparsity: {current_sparsity*100:.2f}\n")
    
            if reinit:
                with torch.no_grad():
                    for p, orig_p in zip(self.weights, original_params):
                        p.copy_(orig_p)
    
            if abs(current_sparsity - sparsity) < 1e-3:
                break
    
        return self.mask
