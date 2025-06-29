#This script is borrowed from https://github.com/alinlab/cs-kd/blob/master/datasets.py
import csv, torchvision, numpy as np, random, os
from PIL import Image
import sys

sys.path.append('/content/Dynamic-Neural-Regeneration/DNR')

from torch.utils.data import Sampler, Dataset, DataLoader, BatchSampler, SequentialSampler, RandomSampler, Subset
from torchvision import transforms, datasets
from collections import defaultdict
import math
import itertools
from utils import augmentations as aug
from data.Dataloader_analysis.cifar10_noisy import CIFAR10ImbalancedNoisy
from data.Dataloader_analysis.cifar100_noisy import CIFAR100ImbalancedNoisy
# from data.Dataloader_analysis.tiny_imgenet_noisy import TinyImageNet_noisy
from datasets import load_dataset  #

from configs.base_config import Config

cfg = Config().parse(None)


class PairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k*self.batch_size
                batch_indices = indices[offset:offset+self.batch_size]
            else:
                batch_indices = random.sample(range(len(self.dataset)),
                                              self.batch_size)

            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.get_class(idx)
                pair_indices.append(random.choice(self.dataset.classwise_indices[y]))

            yield batch_indices + pair_indices
#             yield list(itertools.chain(*zip(batch_indices,pair_indices )))

    def __len__(self):
        if self.num_iterations is None:
            return (len(self.dataset)+self.batch_size-1) // self.batch_size
        else:
            return self.num_iterations


class DatasetWrapper(Dataset):
    # Additinoal attributes
    # - indices
    # - classwise_indices
    # - num_classes
    # - get_class

    def __init__(self, dataset, indices=None):
        self.base_dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices

        # torchvision 0.2.0 compatibility
        if torchvision.__version__.startswith('0.2'):
            if isinstance(self.base_dataset, datasets.ImageFolder):
                self.base_dataset.targets = [s[1] for s in self.base_dataset.imgs]
            else:
                if self.base_dataset.train:
                    self.base_dataset.targets = self.base_dataset.train_labels
                else:
                    self.base_dataset.targets = self.base_dataset.test_labels

        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.targets[self.indices[i]]
            self.classwise_indices[y].append(i)
        self.num_classes = max(self.classwise_indices.keys())+1        

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.targets[self.indices[i]]


class ConcatWrapper(Dataset): # TODO: Naming
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    @staticmethod
    def numcls(sequence):
        s = 0
        for e in sequence:
            l = e.num_classes
            s += l
        return s

    @staticmethod
    def clsidx(sequence):
        r, s, n = defaultdict(list), 0, 0
        for e in sequence:
            l = e.classwise_indices
            for c in range(s, s + e.num_classes):
                t = np.asarray(l[c-s]) + n
                r[c] = t.tolist()
            s += e.num_classes
            n += len(e)
        return r

    def __init__(self, datasets):
        super(ConcatWrapper, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        # for d in self.datasets:
        #     assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

        self.num_classes = self.numcls(self.datasets)
        self.classwise_indices = self.clsidx(self.datasets)
     

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def get_class(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        true_class = self.datasets[dataset_idx].base_dataset.targets[self.datasets[dataset_idx].indices[sample_idx]]
        return self.datasets[dataset_idx].base_dataset.target_transform(true_class)

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes



def load_dataset(name, root, sample='default', **kwargs):
    # Dataset
    if name in ['imagenet', 'tinyImagenet_full', 'tinyImagenet_val', 'CUB200', 'STANFORD120', 'MIT67', 'Aircrafts', 'Dog120', 'Flower102','CUB200_val', 'Dog120_val', 'MIT67_val']:
        # TODO
        if name in ['tinyImagenet_full', 'tinyImagenet_val']:

            transform_train = aug.TrainTransform_tinyimagenet()
            # transform_train = transforms.Compose([
            #     transforms.RandomCrop(64, padding=4),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            # ])
            transform_test = transforms.Compose([
                # transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            train_val_dataset_dir = os.path.join(root, name, "train")
            # print(train_val_dataset_dir)
            test_dataset_dir = os.path.join(root, name, "test")

            trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train))
            valset   = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))

        elif name == 'imagenet':
            transform_train = aug.TrainTransform_imagenet()
            # transform_train = transforms.Compose([
            #     transforms.RandomResizedCrop(224),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            # ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            train_val_dataset_dir = os.path.join(root, "train")
            test_dataset_dir = os.path.join(root, "val")

            trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train))
            valset   = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))

        else:
            transform_train = aug.TrainTransform_other()
            # transform_train = transforms.Compose([
            #     transforms.Resize((256, 256)),
            #     transforms.RandomResizedCrop(224),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            # ])
            transform_test = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            train_val_dataset_dir = os.path.join(root, name, "train")
            trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train))
            
            if name in ['Aircrafts', 'Flower102']:
                val_dataset_dir = os.path.join(root, name, "val")
                test_dataset_dir = os.path.join(root, name, "test")
                valset   = DatasetWrapper(datasets.ImageFolder(root=val_dataset_dir, transform=transform_test))
                testset   = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))

            else:
                test_dataset_dir = os.path.join(root, name, "test")
                valset   = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))

    elif name in ['CIFAR10', 'CIFAR10val', 'CIFAR100', 'CIFAR100val']:
        # transform_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                          (0.2023, 0.1994, 0.2010)),
        # ])
        transform_train = aug.TrainTransform_cifar()
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # if name == 'cifar10':
        #     CIFAR = datasets.CIFAR10
        # else:
        #     CIFAR = datasets.CIFAR100

        # trainset = DatasetWrapper(CIFAR(root, train=True,  download=True, transform=transform_train))
        # valset   = DatasetWrapper(CIFAR(root, train=False, download=True, transform=transform_test))

        train_val_dataset_dir = os.path.join(root, name, "train")
        trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train))
        test_dataset_dir = os.path.join(root, name, "test")
        valset   = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))
    else:
        raise Exception('Unknown dataset: {}'.format(name))

    # Sampler
    if sample == 'default':
        get_train_sampler = lambda d: BatchSampler(RandomSampler(d), kwargs['batch_size'], False)
        get_test_sampler  = lambda d: BatchSampler(SequentialSampler(d), kwargs['batch_size'], False)

    elif sample == 'pair':
        get_train_sampler = lambda d: PairBatchSampler(d, kwargs['batch_size'])
        get_test_sampler  = lambda d: BatchSampler(SequentialSampler(d), kwargs['batch_size'], False)

    else:
        raise Exception('Unknown sampling: {}'.format(sampling))

    trainloader = DataLoader(trainset, batch_sampler=get_train_sampler(trainset), num_workers=10, pin_memory=True)
    valloader   = DataLoader(valset,   batch_sampler=get_test_sampler(valset), num_workers=10, pin_memory=True)

    epoch_size = len(trainset)
    trainloader.num_batches = math.ceil(epoch_size / kwargs['batch_size'])
    trainloader.num_files = epoch_size    
    
    epoch_size = len(valset)
    valloader.num_batches = math.ceil(epoch_size / kwargs['batch_size'])
    valloader.num_files = epoch_size        
    
    if name in ['Aircrafts', 'Flower102']:
        testloader   = DataLoader(testset,   batch_sampler=get_test_sampler(testset), num_workers=10, pin_memory=True)
        epoch_size = len(testset)
        testloader.num_batches = math.ceil(epoch_size / kwargs['batch_size'])
        testloader.num_files = epoch_size   
        return trainloader, valloader, testloader
    else:
        return trainloader, valloader


def load_dataset_linear_eval(name, root, sample='default', **kwargs):
    # Dataset
    if name in ['imagenet', 'tinyImagenet_full', 'tinyImagenet_val', 'CUB200', 'STANFORD120', 'MIT67', 'Aircrafts',
                'Dog120', 'Flower102', 'CUB200_val', 'Dog120_val', 'MIT67_val']:
        # TODO
        if name in ['tinyImagenet_full', 'tinyImagenet_val']:
            transform_train = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            transform_test = transforms.Compose([
                # transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            train_val_dataset_dir = os.path.join(root,  "train")
            test_dataset_dir = os.path.join(root,  "val")
            if cfg.noisy_labels or cfg.class_imbalance:
                if cfg.class_imbalance:
                    class_imb_gamma = 1
                    trainset = TinyImageNet_noisy(
                        root=train_val_dataset_dir,
                        train=True,
                        num_classes=200,
                        perc=1.0,
                        gamma=class_imb_gamma,
                        corrupt_prob=cfg.corrup_prob,
                        transform=transform_train,
                    )
                elif cfg.noisy_labels:
                    class_imb_gamma = -1
                    trainset = CIFAR100ImbalancedNoisy(
                        root=train_val_dataset_dir,
                        train=True,
                        download=False,
                        num_classes=100,
                        perc=1.0,
                        gamma=class_imb_gamma,
                        n_max=int(5000 * 1.0),
                        n_min=int(250 * 1.0),
                        corrupt_prob=cfg.corrup_prob,
                        transform=transform_train,
                    )
            else:
                trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train))

            if cfg.adversarial_attack:
                transform_nonormalize = transforms.Compose([
                    # transforms.Resize(32),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                valset = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_nonormalize))
            else:
                valset = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))

        elif name == 'imagenet':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            train_val_dataset_dir = os.path.join(root, "train")
            test_dataset_dir = os.path.join(root, "val")

            trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train))
            valset = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))

        else:
            transform_train = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            transform_test = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            transform_train_blur = transforms.Compose([
                transforms.Resize((56, 56)),
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            transform_nonormalize = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            train_val_dataset_dir = os.path.join(root, name, "train")
            trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train))
            # trainset_deficit = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train_blur))

            sample_subset = list(range(0, len(trainset), 5))
            # trainset_subset = Subset(trainset, sample_subset)


            if name in ['Aircrafts', 'Flower102']:
                val_dataset_dir = os.path.join(root, name, "test")
                test_dataset_dir = os.path.join(root, name, "test")
                if cfg.adversarial_attack:

                    testset = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_nonormalize))
                    valset = DatasetWrapper(datasets.ImageFolder(root=val_dataset_dir, transform=transform_nonormalize))

                else:
                    testset = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))
                    valset = DatasetWrapper(datasets.ImageFolder(root=val_dataset_dir, transform=transform_test))

                # valset = DatasetWrapper(datasets.ImageFolder(root=val_dataset_dir, transform=transform_test))
                # testset = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))

            else:
                if cfg.adversarial_attack:
                    test_dataset_dir = os.path.join(root, name, "test")
                    valset = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_nonormalize))
                else:
                    test_dataset_dir = os.path.join(root, name, "test")
                    valset = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))

    elif name in ['CIFAR10', 'CIFAR10val', 'CIFAR100', 'CIFAR100val']:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_nonormalize = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_val_dataset_dir = os.path.join(root, name, "train")


        if cfg.noisy_labels or cfg.class_imbalance :
            if name=='CIFAR10':
                if cfg.class_imbalance:
                    class_imb_gamma =1
                    trainset = CIFAR10ImbalancedNoisy(
                        root=os.path.join(root, name),
                        train=True,
                        download=True,
                        num_classes=10,
                        perc=1.0,
                        gamma=class_imb_gamma,
                        n_max=int(5000 * 1.0),
                        n_min=int(250 * 1.0),
                        corrupt_prob=cfg.corrup_prob,
                        transform=transform_train,
                    )
                elif cfg.noisy_labels:
                    class_imb_gamma = -1
                    trainset = CIFAR10ImbalancedNoisy(
                        root=os.path.join(root, name),
                        train=True,
                        download=True,
                        num_classes=10,
                        perc=1.0,
                        gamma=class_imb_gamma,
                        n_max=int(5000 * 1.0),
                        n_min=int(250 * 1.0),
                        corrupt_prob=cfg.corrup_prob,
                        transform=transform_train,
                    )
            elif name=='CIFAR100':
                if cfg.class_imbalance:
                    class_imb_gamma = 1
                    trainset = CIFAR100ImbalancedNoisy(
                        root=os.path.join(root, name),
                        train=True,
                        download=True,
                        num_classes=100,
                        perc=1.0,
                        gamma=class_imb_gamma,
                        n_max=int(5000 * 1.0),
                        n_min=int(250 * 1.0),
                        corrupt_prob=cfg.corrup_prob,
                        transform=transform_train,
                    )
                elif cfg.noisy_labels:
                    class_imb_gamma = -1
                    trainset = CIFAR100ImbalancedNoisy(
                        root=os.path.join(root, name),
                        train=True,
                        download=True,
                        num_classes=100,
                        perc=1.0,
                        gamma=class_imb_gamma,
                        n_max=int(5000 * 1.0),
                        n_min=int(250 * 1.0),
                        corrupt_prob=cfg.corrup_prob,
                        transform=transform_train,
                    )
        else:
            trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train))


        if cfg.adversarial_attack:
            test_dataset_dir = os.path.join(root, name, "test")
            valset = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_nonormalize))
        else:
            test_dataset_dir = os.path.join(root, name, "test")
            valset = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))
        # valset = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))
    else:
        raise Exception('Unknown dataset: {}'.format(name))

    # Sampler
    if sample == 'default':
        get_train_sampler = lambda d: BatchSampler(RandomSampler(d), kwargs['batch_size'], False)
        get_test_sampler = lambda d: BatchSampler(SequentialSampler(d), kwargs['batch_size'], False)

    elif sample == 'pair':
        get_train_sampler = lambda d: PairBatchSampler(d, kwargs['batch_size'])
        get_test_sampler = lambda d: BatchSampler(SequentialSampler(d), kwargs['batch_size'], False)

    else:
        raise Exception('Unknown sampling: {}'.format(sampling))

    trainloader = DataLoader(trainset, batch_sampler=get_train_sampler(trainset), num_workers=4, pin_memory=True)
    valloader = DataLoader(valset, batch_sampler=get_test_sampler(valset), num_workers=4, pin_memory=True)
    # train_subloader = DataLoader(trainset_subset, batch_sampler=get_test_sampler(trainset_subset), num_workers=4, pin_memory=True)
    # trainloader_deficit = DataLoader(trainset_deficit, batch_sampler=get_train_sampler(trainset_deficit), num_workers=4, pin_memory=True)


    epoch_size = len(trainset)
    trainloader.num_batches = math.ceil(epoch_size / kwargs['batch_size'])
    trainloader.num_files = epoch_size
    # trainloader_deficit.num_batches = math.ceil(epoch_size / kwargs['batch_size'])
    # trainloader_deficit.num_files = epoch_size
    epoch_size = len(valset)
    valloader.num_batches = math.ceil(epoch_size / kwargs['batch_size'])
    valloader.num_files = epoch_size

    if name in ['Aircrafts', 'Flower102']:
        testloader = DataLoader(testset, batch_sampler=get_test_sampler(testset), num_workers=10, pin_memory=True)
        epoch_size = len(testset)
        testloader.num_batches = math.ceil(epoch_size / kwargs['batch_size'])
        testloader.num_files = epoch_size
        return trainloader, valloader, testloader
    else:
        return trainloader, valloader #, train_subloader, trainloader_deficit
