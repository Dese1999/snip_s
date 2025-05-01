
from PIL import Image
import os
import os.path
import numpy as np

def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist, sep):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split(sep)
            imlist.append((impath, int(imlabel)))

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader, sep=' '):
        self.root = root
        self.imlist = flist_reader(flist, sep)
        self.targets = np.array([datapoint[1] for datapoint in self.imlist])
        self.data = np.array([datapoint[0] for datapoint in self.imlist])
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)


class TinyImageNet_noisy(ImageFilelist):
    def __init__(self, gamma=-1, n_min=25, n_max=500, num_classes=200, perc=1.0, **kwargs):
        super(TinyImageNet_noisy, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.perc = perc
        self.gamma = gamma
        self.n_min = n_min
        self.n_max = n_max
        self.n_max = n_max

        if perc < 1.0:
            print('*' * 30)
            print('Creating a Subset of Dataset')
            self.get_subset()
            (unique, counts) = np.unique(self.targets, return_counts=True)
            frequencies = np.asarray((unique, counts)).T
            print(frequencies)

        if gamma > 0:
            print('*' * 30)
            print('Creating Imbalanced Dataset')
            self.imbalanced_dataset()
            (unique, counts) = np.unique(self.targets, return_counts=True)
            frequencies = np.asarray((unique, counts)).T
            print(frequencies)

    def get_subset(self):
        np.random.seed(12345)

        lst_data = []
        lst_targets = []
        targets = np.array(self.targets)
        for class_idx in range(self.num_classes):
            class_indices = np.where(targets == class_idx)[0]
            num_samples = int(self.perc * len(class_indices))
            sel_class_indices = class_indices[:num_samples]
            lst_data.append(self.data[sel_class_indices])
            lst_targets.append(targets[sel_class_indices])

        self.data = np.concatenate(lst_data)
        self.targets = np.concatenate(lst_targets)

        self.imlist = list(zip(self.data.tolist(), self.targets.tolist()))

        assert len(self.targets) == len(self.data)

    def imbalanced_dataset(self):
        np.random.seed(12345)
        X = np.array([[1, -self.n_max], [1, -self.n_min]])
        Y = np.array([self.n_max, self.n_min * self.num_classes ** (self.gamma)])

        a, b = np.linalg.solve(X, Y)

        classes = list(range(1, self.num_classes + 1))

        imbal_class_counts = []
        for c in classes:
          num_c = int(np.round(a / (b + (c) ** (self.gamma))))
          # print(c, num_c)
          imbal_class_counts.append(num_c)

        targets = np.array(self.targets)

        # Get class indices
        class_indices = [np.where(targets == i)[0] for i in range(self.num_classes)]

        # Get imbalanced number of instances
        imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
        imbal_class_indices = np.hstack(imbal_class_indices)

        np.random.shuffle(imbal_class_indices)

        # Set target and data to dataset
        self.targets = targets[imbal_class_indices]
        self.data = self.data[imbal_class_indices]

        self.imlist = list(zip(self.data.tolist(), self.targets.tolist()))

        assert len(self.targets) == len(self.data)


class TinyImageNetRandomSubset(ImageFilelist):
    def __init__(self, corrupt_prob=0.0, num_classes=200, **kwargs):
        super(TinyImageNetRandomSubset, self).__init__(**kwargs)
        self.n_classes = num_classes
        labels = np.array(self.targets)
        self.data = self.data[labels < num_classes]
        labels = labels[labels < num_classes]
        labels = [int(x) for x in labels]
        self.targets = labels
        print('Number of Training Samples:', len(self.targets))

        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):

        labels = np.array(self.targets)

        print('Original Labels:', labels)

        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels

        print('Noisy Labels:', labels)

        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]
        self.targets = labels

        self.imlist = list(zip(self.data.tolist(), self.targets))
        assert len(self.targets) == len(self.data)


# dataset_path = r'/data/input/datasets/tiny_imagenet/tiny-imagenet-200'
# dataset = TinyImageNetRandomSubset(
#             root=dataset_path,
#             flist=os.path.join(dataset_path, "train_kv_list.txt"),
#             corrupt_prob=1,
#             num_classes=2
# )