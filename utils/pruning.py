import matplotlib.pyplot as plt
from torchvision import *
import numpy as np
import torch
from copy import deepcopy
import utils
from utils.net_utils import create_dense_mask_0
import torch.nn as nn
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
            # Change applied: Add allow_unused=True and handle None
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
        
        # Calculate current sparsity
        current_sparsity = sum(self.pruned) / sum([ind.numel() for ind in self.indicators])
        
        if not silent:
            print("weights left: ", [self.indicators[i].numel() - pruned for i, pruned in enumerate(self.pruned)])
            print("sparsities: ", [round(100 * pruned / self.indicators[i].numel(), 2) for i, pruned in enumerate(self.pruned)])
            print(f"Current total sparsity: {current_sparsity*100:.2f}\n")
        
        return self.mask_

    def snip_it(self, sparsity, steps=5, mini_batches=1, silent=False):
        """
        Apply SNIP-it pruning (iterative, unstructured) based on elasticity criterion.
        Args:
            sparsity: Target sparsity level (0 to 1).
            steps: Number of pruning steps.
            mini_batches: Number of mini-batches to compute gradients.
            silent: If True, suppress printing.
        Returns:
            Updated mask_.
        """
        # Calculate pruning steps: start with 50% and halve the remainder each step
        start = 0.5
        prune_steps = [sparsity - (sparsity - start) * (0.5 ** i) for i in range(steps)] + [sparsity]
        print(f"prune_steps: {prune_steps}")
        for step, target_sparsity in enumerate(prune_steps):
            if not silent:
                print(f"SNIP-it step {step + 1}/{len(prune_steps)}: Targeting sparsity {target_sparsity:.4f}")
            
            # Compute gradients
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
            
            loss /= max(1, mini_batch)  # Average loss for elasticity
            
            with torch.no_grad():
                # Compute weight-elasticity (Formula 3: |grad * weight| / loss)
                saliences = [(grad * weight).view(-1).abs() / (loss + 1e-8) 
                             for grad, weight in zip(grads, self.weights)]
                saliences = torch.cat(saliences).cpu()
                thresh = float(saliences.kthvalue(int(target_sparsity * saliences.shape[0]))[0])
                
                for j, (grad, weight, layer) in enumerate(zip(grads, self.weights, self.indicators)):
                    layer[(grad * weight).abs() / (loss + 1e-8) <= thresh] = 0
                    self.pruned[j] = int(torch.sum(layer == 0))
            
            self.indicate()
            
            # Update sparsity metrics
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
        """
        Apply SNAP-it pruning (iterative, structured) based on node-elasticity criterion.
        Args:
            sparsity: Target sparsity level (0 to 1).
            steps: Number of pruning steps.
            start: Starting sparsity for the pruning schedule.
            mini_batches: Number of mini-batches to compute gradients.
            silent: If True, suppress printing.
        Returns:
            Updated mask_.
        """
        # Calculate pruning steps
        prune_steps = [sparsity - (sparsity - start) * (0.5 ** i) for i in range(steps)] + [sparsity]
        current_sparsity = 0.0
        remaining = 1.0
        
        for step, target_sparsity in enumerate(prune_steps):
            if not silent:
                print(f"SNAP-it step {step + 1}/{len(prune_steps)}: Targeting sparsity {target_sparsity:.4f}")
            
            # Calculate pruning rate for this step
            prune_rate = (target_sparsity - current_sparsity) / (remaining + 1e-8)
            
            # Compute gradients
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
            
            loss /= max(1, mini_batch)  # Average loss for elasticity
            
            with torch.no_grad():
                # Compute node-elasticity for Conv2d layers (Formula 4)
                saliences = []
                layer_names = [name for name, _ in self.model.named_parameters() if 'mask' not in name]
                for j, (grad, weight, layer) in enumerate(zip(grads, self.weights, self.indicators)):
                    if len(weight.shape) == 4:  # Conv2d layer
                        importance = torch.sum(grad.abs(), dim=(1, 2, 3)) / (loss + 1e-8)  # Node-elasticity
                        #if not silent:
                           # print(f"Channel importance for {layer_names[j]}: {importance.cpu().numpy()}")
                        saliences.append(importance.view(-1))
                    else:
                        # For non-Conv2d layers, use weight-elasticity
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
            
            # Update sparsity metrics
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
                L = nn.CrossEntropyLoss()(output, y)
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
        """
        Apply SNIP-R pruning (perturbation-based).
        """
        with torch.no_grad():
            saliences = [torch.zeros_like(w) for w in self.weights]
            x, y = next(iter(self.loader))
            x, y = x.to(self.device), y.to(self.device)
            z = self.model.forward(x)
            L0 = torch.nn.CrossEntropyLoss()(z, y)  # Baseline loss

            for laynum, layer in enumerate(self.weights):
                if not silent:
                    print("layer ", laynum, "...")
                for weight in range(layer.numel()):
                    temp = layer.view(-1)[weight].clone()
                    layer.view(-1)[weight] = 0

                    z = self.model.forward(x)  # Forward pass
                    L = torch.nn.CrossEntropyLoss()(z, y)  # Loss
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
        """
        Compute importance based on weight and gradient magnitudes.
        """
        mask = utils.net_utils.create_dense_mask_0(deepcopy(self.model), device, value=0)
        for (name, param), param_mask in zip(self.model.named_parameters(), mask.parameters()):
            if 'weight' in name and 'bn' not in name and 'downsample' not in name:
                param_mask.data = abs(param.data) + abs(param.grad)

        imp = [layer for name, layer in mask.named_parameters() if 'mask' not in name]
        imp = torch.cat([i.view(-1).cpu() for i in imp])
        percentile = np.percentile(imp.numpy(), sparsity * 100)  # Get a value for this percentile
        above_threshold = [i > percentile for i in imp]
        for i, param_mask in enumerate(mask.parameters()):
            param_mask.data = param_mask.data * above_threshold[i].view(param_mask.shape).to(device)
        return mask

    def apply_reg(self, mask):
        """
        Apply regularization based on mask.
        """
        for (name, param), param_mask in zip(self.model.named_parameters(), mask.parameters()):
            if 'weight' in name and 'bn' not in name and 'downsample' not in name:
                l2_grad = param_mask.data * param.data
                param.grad += l2_grad

    def update_reg(self, mask, reg_decay, cfg):
        """
        Update regularization mask based on configuration.
        """
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
