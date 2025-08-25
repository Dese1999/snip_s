import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
import numpy as np
from PIL import ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from .datasets import DatasetWrapper, PairBatchSampler

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

from torch.utils.data import Subset

# class CIFAR10val:
#     def __init__(self, cfg):
#         data_dir = cfg.data
#         batch_size = cfg.batch_size
#         num_workers = cfg.num_threads
#         reduced_train_ratio = 0.1

#         transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#             transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
#         ])
#         transform_val_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ])

#         full_trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
#         val_size = int(0.1 * len(full_trainset))
#         train_size = len(full_trainset) - val_size
#         indices = torch.randperm(len(full_trainset)).tolist()
#         train_indices = indices[:train_size]
#         val_indices = indices[train_size:]
#         reduced_train_size = int(train_size * reduced_train_ratio)
#         train_indices = train_indices[:reduced_train_size]

#         trainset = Subset(full_trainset, train_indices)
#         valset = Subset(full_trainset, val_indices)
#         testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_val_test)

#         self.train_loader = DataLoader(
#             trainset, batch_size=batch_size, shuffle=True,
#             num_workers=num_workers, pin_memory=True)
#         self.val_loader = DataLoader(
#             valset, batch_size=batch_size, shuffle=False,
#             num_workers=num_workers, pin_memory=True)
#         self.tst_loader = DataLoader(
#             testset, batch_size=batch_size, shuffle=False,
#             num_workers=num_workers, pin_memory=True)
#         self.num_classes = 10

#         print(f"Training samples count: {len(trainset)}")
#         print(f"Validation samples count: {len(valset)}")
#         print(f"Test samples count: {len(testset)}")



class CIFAR10val:
    def __init__(self, cfg):
        """
        Initialize CIFAR10val dataset with support for both cs_kd=True and cs_kd=False.
        
        Args:
            cfg: Configuration object containing data_dir, batch_size, num_threads, cs_kd, etc.
        """
        data_dir = cfg.data
        batch_size = cfg.batch_size
        num_workers = cfg.num_threads
        reduced_train_ratio = 0.1

        # Define transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
        ])
        transform_val_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Load full CIFAR10 dataset
        full_trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        
        # Split indices for train and validation
        val_size = int(0.1 * len(full_trainset))  # 10% for validation
        train_size = len(full_trainset) - val_size
        indices = torch.randperm(len(full_trainset)).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        reduced_train_size = int(train_size * reduced_train_ratio)  # 10% of train_size
        train_indices = train_indices[:reduced_train_size]

        # Validate indices
        assert max(train_indices) < len(full_trainset), f"Train indices out of range: max={max(train_indices)}, dataset length={len(full_trainset)}"
        assert max(val_indices) < len(full_trainset), f"Validation indices out of range: max={max(val_indices)}, dataset length={len(full_trainset)}"

        # Choose between Subset and DatasetWrapper based on cs_kd
        if cfg.cs_kd:
            trainset = DatasetWrapper(full_trainset, indices=train_indices)
            valset = DatasetWrapper(full_trainset, indices=val_indices)
            train_sampler = PairBatchSampler(trainset, batch_size, num_iterations=None)
            self.train_loader = DataLoader(
                trainset, shuffle=False, batch_sampler=train_sampler,
                num_workers=num_workers, pin_memory=True)
        else:
            trainset = Subset(full_trainset, train_indices)
            valset = Subset(full_trainset, val_indices)
            self.train_loader = DataLoader(
                trainset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True, drop_last=True)

        # Validation and test DataLoaders
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_val_test)
        self.val_loader = DataLoader(
            valset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
        self.tst_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
        self.num_classes = 10

        # Log dataset sizes
        print(f"Training samples count: {len(trainset)}")
        print(f"Validation samples count: {len(valset)}")
        print(f"Test samples count: {len(testset)}")
