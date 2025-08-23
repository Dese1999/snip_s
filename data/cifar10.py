import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from PIL import ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from .datasets import PairBatchSampler, DatasetWrapper


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
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
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1) ],p=0.8,),
            #Solarization(p=0.1),
            transforms.ToTensor(),                      # Convert images to PyTorch tensors
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize with CIFAR-10 mean and std
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
            
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
        # Configuration parameters
        data_dir = cfg.data
        batch_size = cfg.batch_size
        num_workers = cfg.num_threads
        reduced_train_ratio = 0.1  # Ratio to reduce training data

        # Training data transformations
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value='random'
            )
        ])

        # Validation and test data transformations
        transform_val_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Load full training dataset
        full_trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train)

        # Wrap dataset with DatasetWrapper
        full_trainset = DatasetWrapper(full_trainset)

        # Split data into train/val
        val_size = int(0.1 * len(full_trainset))  # 5000 samples
        train_size = len(full_trainset) - val_size  # 45000 samples
        indices = torch.randperm(len(full_trainset)).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Reduce training data to 10% (after initial split)
        reduced_train_size = int(train_size * reduced_train_ratio)  # 4500 samples
        train_indices = train_indices[:reduced_train_size]

        # Create reduced trainset and valset
        trainset = DatasetWrapper(full_trainset, indices=train_indices)
        valset = DatasetWrapper(full_trainset, indices=val_indices)

        # Load test data
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_val_test)

        # Create samplers
        sampler_type = "pair" if cfg.cs_kd else "default"
        if sampler_type == "pair":
            train_sampler = PairBatchSampler(trainset, batch_size, num_iterations=None)
        else:
            train_sampler = SubsetRandomSampler(train_indices)

        # Create DataLoaders
        if sampler_type == "pair":
            self.train_loader = DataLoader(
                trainset, shuffle=False,
                batch_sampler=train_sampler, num_workers=num_workers, pin_memory=True)
        else:
            self.train_loader = DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                sampler=train_sampler, num_workers=num_workers, pin_memory=True)

        
        self.val_loader = DataLoader(
            valset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        self.tst_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        self.num_classes = 10

        # Print dataset information
        print(f"Training samples count: {len(trainset)}")
        print(f"Validation samples count: {len(valset)}")
        print(f"Test samples count: {len(testset)}")
