import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

class CIFAR10:
    def __init__(self, cfg):
        # Define data directory, batch size, and number of workers from config
        data_dir = cfg.data
        batch_size = cfg.batch_size if not cfg.eval_linear else cfg.linear_batch_size
        num_workers = cfg.num_threads

        # Define transformations for training data
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),       # Randomly crop images to 32x32 with padding
            transforms.RandomHorizontalFlip(),          # Randomly flip images horizontally
            transforms.ToTensor(),                      # Convert images to PyTorch tensors
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize with CIFAR-10 mean and std
        ])
        
        # Define transformations for test data
        transform_test = transforms.Compose([
            transforms.ToTensor(),                      # Convert images to PyTorch tensors
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize with CIFAR-10 mean and std
        ])

        # Load CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

        # Set sampler type based on config
        sampler_type = "pair" if cfg.cs_kd else "default"
        train_sampler = None  # Add specific implementation for 'pair' sampler if needed

        # Create DataLoaders for training and testing
        self.train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None),
                                       sampler=train_sampler, num_workers=num_workers, pin_memory=True)
        self.tst_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True)
        self.val_loader = self.tst_loader  # Validation loader is same as test loader in this case
        self.num_classes = 10  # Number of classes in CIFAR-10

class CIFAR10val:
    def __init__(self, cfg):
        # Define data directory, batch size, and number of workers from config
        data_dir = cfg.data
        batch_size = cfg.batch_size
        num_workers = cfg.num_threads

        # Define transformations for training data
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),       # Randomly crop images to 32x32 with padding
            transforms.RandomHorizontalFlip(),          # Randomly flip images horizontally
            transforms.ToTensor(),                      # Convert images to PyTorch tensors
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize with CIFAR-10 mean and std
        ])
        
        # Define transformations for validation data
        transform_val = transforms.Compose([
            transforms.ToTensor(),                      # Convert images to PyTorch tensors
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize with CIFAR-10 mean and std
        ])

        # Load CIFAR-10 dataset and split into training and validation sets
        full_trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        val_size = int(0.1 * len(full_trainset))  # 10% of data for validation
        train_size = len(full_trainset) - val_size
        train_indices, val_indices = torch.randperm(len(full_trainset)).split([train_size, val_size])
        
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        valset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_val)

        # Set sampler type based on config
        sampler_type = "pair" if cfg.cs_kd else "default"
        train_sampler = SubsetRandomSampler(train_indices) if sampler_type == "default" else None
        val_sampler = SubsetRandomSampler(val_indices)

        # Create DataLoaders for training and validation
        self.train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False,
                                       sampler=train_sampler, num_workers=num_workers, pin_memory=True)
        self.val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False,
                                     sampler=val_sampler, num_workers=num_workers, pin_memory=True)
        self.tst_loader = self.val_loader  # Test loader is same as validation loader in this case
        self.num_classes = 10  # Number of classes in CIFAR-10
