import numpy as np
import idx2numpy
from torchvision.transforms import v2
from torchvision import models, datasets
import os
import requests
import pickle
import sys
from tqdm import tqdm
import hashlib
from .structures import *
from torch import tensor, load, save, cat, from_numpy, LongTensor, Tensor, float32, no_grad
from torch.nn import Sequential
from collections import defaultdict
from torch.cuda import empty_cache

PATH_MNIST_X_TRAIN = "datasets/MNIST/raw/train-images-idx3-ubyte"
PATH_MNIST_Y_TRAIN = "datasets/MNIST/raw/train-labels-idx1-ubyte"
PATH_MNIST_X_TEST = "datasets/MNIST/raw/t10k-images-idx3-ubyte"
PATH_MNIST_Y_TEST = "datasets/MNIST/raw/t10k-labels-idx1-ubyte"

PATH_FASHION_MNIST_X_TRAIN = "datasets/FashionMNIST/raw/train-images-idx3-ubyte"
PATH_FASHION_MNIST_Y_TRAIN = "datasets/FashionMNIST/raw/train-labels-idx1-ubyte"
PATH_FASHION_MNIST_X_TEST = "datasets/FashionMNIST/raw/t10k-images-idx3-ubyte"
PATH_FASHION_MNIST_Y_TEST = "datasets/FashionMNIST/raw/t10k-labels-idx1-ubyte"

PATH_EMNIST_X_TRAIN = "datasets/EMNIST/raw/emnist-balanced-train-images-idx3-ubyte"
PATH_EMNIST_Y_TRAIN = "datasets/EMNIST/raw/emnist-balanced-train-labels-idx1-ubyte"
PATH_EMNIST_X_TEST = "datasets/EMNIST/raw/emnist-balanced-test-images-idx3-ubyte"
PATH_EMNIST_Y_TEST = "datasets/EMNIST/raw/emnist-balanced-test-labels-idx1-ubyte"

PATH_KMNIST_X_TRAIN = "datasets/KMNIST/raw/train-images-idx3-ubyte"
PATH_KMNIST_Y_TRAIN = "datasets/KMNIST/raw/train-labels-idx1-ubyte"
PATH_KMNIST_X_TEST = "datasets/KMNIST/raw/t10k-images-idx3-ubyte"
PATH_KMNIST_Y_TEST = "datasets/KMNIST/raw/t10k-labels-idx1-ubyte"

PATH_CIFAR10 = "datasets/cifar-10-batches-py"
PATH_CIFAR10_DATABATCH = [
    f"{PATH_CIFAR10}/data_batch_{i}" for i in range(1, 6)]
PATH_CIFAR10_TESTBATCH = f"{PATH_CIFAR10}/test_batch"

PATH_CIFAR100 = "datasets/cifar-100-python"
PATH_CIFAR100_DATABATCH = [f"{PATH_CIFAR100}/train"]
PATH_CIFAR100_TESTBATCH = f"{PATH_CIFAR100}/test"
PATH_CIFAR100_META = f"{PATH_CIFAR100}/meta"

REPOSITORY_CORE50_NPZ_128 = "http://bias.csr.unibo.it/maltoni/download/core50/core50_imgs.npz"
REPOSITORY_CORE50_PATHS = "https://vlomonaco.github.io/core50/data/paths.pkl"
REPOSITORY_CORE50_LABELS = "https://vlomonaco.github.io/core50/data/labels.pkl"
REPOSITORY_CORE50_LUP = "https://vlomonaco.github.io/core50/data/LUP.pkl"


class GPULoading:
    """ Load local datasets on GPU using the GPUTensorDataset

    Args:
        device (str, optional): Device to use. Defaults to "cuda:0".
    """

    def __init__(self, device="cuda:0", root="datasets", *args, **kwargs):
        self.device = device
        self.root = root
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)

    def task_selection(self, task, padding=0, *args, **kwargs):
        """ Select the task to load

        Args:
            task (str): Name of the task

        Returns:
            train (GPUTensorDataset): Training dataset
            test (GPUTensorDataset): Testing dataset
            shape (tuple): Shape of the data
            target_size (int): Number of classes
        """
        self.padding = padding

        if "emnist" in task.lower():
            train, test = self.emnist(*args, **kwargs)
        elif "kmnist" in task.lower():
            train, test = self.kmnist(*args, **kwargs)
        elif "fullpmnist" in task.lower():
            train, test = self.permuted_mnist_full(
                *args, **kwargs)
        elif "mnist" in task.lower() and "fashion" in task.lower():
            train = []
            test = []
            mnist_train, mnist_test = self.mnist(*args, **kwargs)
            fashion_train, fashion_test = self.fashion_mnist(*args, **kwargs)
            train.append(mnist_train)
            train.append(fashion_train)
            test.append(mnist_test)
            test.append(fashion_test)
        elif "mnist" in task.lower():
            train, test = self.mnist(*args, **kwargs)
        elif "fashion" in task.lower():
            train, test = self.fashion_mnist(*args, **kwargs)
        elif "dilcifar100" in task.lower():
            train, test = self.domain_incremental_cifar100(*args, **kwargs)
        elif "cifar100" in task.lower():
            train, test = self.cifar100(*args, **kwargs)
        elif "cifar10" in task.lower():
            train, test = self.cifar10(*args, **kwargs)
        elif "core50" in task.lower():
            scenario = task.split("-")[1]
            train, test = self.core50(
                scenario=scenario, *args, **kwargs)
        if isinstance(train, GPUTensorDataset):
            shape = train.data[0].shape
            target_size = len(train.targets.unique())
        else:
            shape = train[0].data[0].shape
            target_size = len(train[0].targets.unique())
        return train, test, shape, target_size

    def fashion_mnist(self, *args, **kwargs):
        if not os.path.exists(PATH_FASHION_MNIST_X_TRAIN):
            datasets.FashionMNIST("datasets", download=True)
        return self.mnist_like(PATH_FASHION_MNIST_X_TRAIN, PATH_FASHION_MNIST_Y_TRAIN,
                               PATH_FASHION_MNIST_X_TEST, PATH_FASHION_MNIST_Y_TEST, *args, **kwargs)

    def mnist(self, *args, **kwargs):
        if not os.path.exists(PATH_MNIST_X_TRAIN):
            datasets.MNIST("datasets", download=True)
        return self.mnist_like(PATH_MNIST_X_TRAIN, PATH_MNIST_Y_TRAIN,
                               PATH_MNIST_X_TEST, PATH_MNIST_Y_TEST, *args, **kwargs)

    def emnist(self, *args, **kwargs):
        if not os.path.exists(PATH_EMNIST_X_TRAIN):
            datasets.EMNIST("datasets", download=True, split="balanced")
        return self.mnist_like(PATH_EMNIST_X_TRAIN, PATH_EMNIST_Y_TRAIN,
                               PATH_EMNIST_X_TEST, PATH_EMNIST_Y_TEST, *args, **kwargs)

    def kmnist(self, *args, **kwargs):
        if not os.path.exists(PATH_KMNIST_X_TRAIN):
            datasets.KMNIST("datasets", download=True)
        return self.mnist_like(PATH_KMNIST_X_TRAIN, PATH_KMNIST_Y_TRAIN,
                               PATH_KMNIST_X_TEST, PATH_KMNIST_Y_TEST, *args, **kwargs)

    def mnist_like(self, path_train_x, path_train_y, path_test_x, path_test_y, *args, **kwargs):
        """ Load a local dataset on GPU corresponding either to MNIST or FashionMNIST

        Args:
            batch_size (int): Batch size
            path_train_x (str): Path to the training data
            path_train_y (str): Path to the training labels
            path_test_x (str): Path to the testing data
            path_test_y (str): Path to the testing labels
        """
        # load ubyte dataset
        train_x = idx2numpy.convert_from_file(
            path_train_x).astype(np.float32)
        train_y = idx2numpy.convert_from_file(
            path_train_y).astype(np.float32)
        test_x = idx2numpy.convert_from_file(
            path_test_x).astype(np.float32)
        test_y = idx2numpy.convert_from_file(
            path_test_y).astype(np.float32)
        # Normalize and pad the data
        train_x, test_x = self.normalization(train_x, test_x)
        return self.to_dataset(train_x, train_y, test_x, test_y)

    def permuted_mnist_full(self, *args, **kwargs):
        if not os.path.exists(PATH_MNIST_X_TRAIN):
            datasets.MNIST("datasets", download=True)
        n_tasks = 10
        train_dataset, test_dataset = self.mnist_like(PATH_MNIST_X_TRAIN, PATH_MNIST_Y_TRAIN,
                                                      PATH_MNIST_X_TEST, PATH_MNIST_Y_TEST, *args, **kwargs)
        permutations = [torch.randperm(784).cpu() for _ in range(n_tasks)]
        # create a dataset with n tasks all blended together
        train_x, train_y = train_dataset.data, train_dataset.targets
        test_x, test_y = test_dataset.data, test_dataset.targets
        test_data, test_labels, train_data, train_labels = [], [], [], []
        for i in range(n_tasks):
            perm = permutations[i]
            train_x_new = train_x.view(-1,
                                       784)[:, perm].view(-1, 1, 28, 28).clone()
            test_x_new = test_x.view(-1,
                                     784)[:, perm].view(-1, 1, 28, 28).clone()
            train_data.append(train_x_new)
            test_data.append(test_x_new)
            train_labels.append(train_y)
            test_labels.append(test_y)
        train_data = cat(train_data)
        test_data = cat(test_data)
        train_labels = cat(train_labels)
        test_labels = cat(test_labels)
        return train_dataset, test_dataset

    def cifar10(self, iterations=10, *args, **kwargs):
        """ Load a local dataset on GPU corresponding to CIFAR10 """
        # Deal with the training data
        if not os.path.exists("datasets/CIFAR10/raw"):
            datasets.CIFAR10("datasets", download=True)
        path_databatch = PATH_CIFAR10_DATABATCH
        path_testbatch = PATH_CIFAR10_TESTBATCH
        if "feature_extraction" in kwargs and kwargs["feature_extraction"] == True:
            folder = "datasets/cifar10_resnet18"
            os.makedirs(folder, exist_ok=True)
            if not os.listdir(folder) or not os.path.exists(f"{folder}/cifar10_{iterations}_features_train.pt"):
                train_x = []
                train_y = []
                for path in path_databatch:
                    with open(path, 'rb') as f:
                        dict = pickle.load(f, encoding='bytes')
                    train_x.append(dict[b'data'])
                    train_y.append(dict[b'labels'])
                train_x = np.concatenate(train_x)
                train_y = np.concatenate(train_y)
                # Deal with the test data
                with open(path_testbatch, 'rb') as f:
                    dict = pickle.load(f, encoding='bytes')
                test_x = dict[b'data']
                test_y = dict[b'labels']
                # Deflatten the data
                train_x = train_x.reshape(-1, 3, 32, 32)
                test_x = test_x.reshape(-1, 3, 32, 32)
                self.feature_extraction(
                    folder, train_x, train_y, test_x, test_y, task="cifar10", iterations=iterations)
            train_x = load(
                f"{folder}/cifar10_{iterations}_features_train.pt")
            train_y = load(
                f"{folder}/cifar10_{iterations}_target_train.pt")
            test_x = load(
                f"{folder}/cifar10_{iterations}_features_test.pt")
            test_y = load(
                f"{folder}/cifar10_{iterations}_target_test.pt")
        else:
            train_x = []
            train_y = []
            for path in path_databatch:
                with open(path, 'rb') as f:
                    dict = pickle.load(f, encoding='bytes')
                train_x.append(dict[b'data'])
                train_y.append(dict[b'labels'])
            train_x = np.concatenate(train_x)
            train_y = np.concatenate(train_y)
            # Deal with the test data
            with open(path_testbatch, 'rb') as f:
                dict = pickle.load(f, encoding='bytes')
            test_x = dict[b'data']
            test_y = dict[b'labels']
            # Deflatten the data
            train_x = train_x.reshape(-1, 3, 32, 32)
            test_x = test_x.reshape(-1, 3, 32, 32)
            # Normalize and pad the data
            train_x, test_x = self.normalization(train_x, test_x)
        return self.to_dataset(train_x, train_y, test_x, test_y)

    def cifar100(self, iterations=10, *args, **kwargs):
        """ Load a local dataset on GPU corresponding to CIFAR100 """
        if not os.path.exists("datasets/CIFAR100/raw"):
            datasets.CIFAR100("datasets", download=True)
        if "feature_extraction" in kwargs and kwargs["feature_extraction"] == True:
            folder = "datasets/cifar100_resnet18"
            os.makedirs(folder, exist_ok=True)
            if not os.listdir(folder) or not os.path.exists(f"{folder}/cifar100_{iterations}_features_train.pt"):
                path_databatch = PATH_CIFAR100_DATABATCH
                path_testbatch = PATH_CIFAR100_TESTBATCH
                with open(path_databatch[0], "rb") as f:
                    data = pickle.load(f, encoding="bytes")
                    train_x = data[b"data"]
                    train_y = data[b"fine_labels"]
                with open(path_testbatch, "rb") as f:
                    data = pickle.load(f, encoding="bytes")
                    test_x = data[b"data"]
                    test_y = data[b"fine_labels"]
                train_x = train_x.reshape(-1, 3, 32, 32)
                test_x = test_x.reshape(-1, 3, 32, 32)
                self.feature_extraction(
                    folder, train_x, train_y, test_x, test_y, task="cifar100", iterations=iterations)
            train_x = load(
                f"{folder}/cifar100_{iterations}_features_train.pt")
            train_y = load(
                f"{folder}/cifar100_{iterations}_target_train.pt")
            test_x = load(
                f"{folder}/cifar100_{iterations}_features_test.pt")
            test_y = load(
                f"{folder}/cifar100_{iterations}_target_test.pt")
        else:
            train_x, fine_labels, test_x, test_fine_labels, coarse_labels, test_coarse_labels = self.read_cifar100()
            train_x = train_x.reshape(-1, 3, 32, 32)
            test_x = test_x.reshape(-1, 3, 32, 32)
            train_y = fine_labels
            test_y = test_fine_labels
            # Normalize and pad the data
            train_x, test_x = self.normalization(train_x, test_x)
        return self.to_dataset(train_x, train_y, test_x, test_y)

    def feature_extraction(self, folder, train_x, train_y, test_x, test_y, task="cifar100", iterations=10):
        """ Extract features using a resnet18 model

        Args:
            folder (str): Folder to save the features
            train_x (tensor): Training data
            train_y (tensor): Training labels
            test_x (tensor): Testing data
            test_y (tensor): Testing labels
            task (str, optional): Name of the task. Defaults to "cifar100".
            iterations (int, optional): Number of passes to make. Defaults to 10.
        """
        print(f"Extracting features from {task}...")
        resnet18 = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        # Remove the classification layer
        resnet18 = Sequential(
            *list(resnet18.children())[:-1])
        # Freeze the weights of the feature extractor
        for param in resnet18.parameters():
            param.requires_grad = False
        # Transforms to apply to augment the data
        transform_train = v2.Compose([
            v2.feature_extraction(220, antialias=True),
            v2.RandomHorizontalFlip(),
        ])
        transform_test = v2.Compose([
            v2.feature_extraction(220, antialias=True),
        ])
        # Extract the features
        features_train = []
        target_train = []
        features_test = []
        target_test = []
        # Normalize
        train_x = from_numpy(train_x).float() / 255
        test_x = from_numpy(test_x).float() / 255
        if len(train_x.size()) == 3:
            train_x = train_x.unsqueeze(1)
            test_x = test_x.unsqueeze(1)
        # Converting the data to a GPU TensorDataset (allows to load everything in the GPU memory at once)
        train_dataset = GPUTensorDataset(
            train_x, Tensor(train_y).type(
                LongTensor), device=self.device)
        test_dataset = GPUTensorDataset(test_x, Tensor(test_y).type(
            LongTensor), device=self.device)
        train_dataset = GPUDataLoader(
            train_dataset, batch_size=1024, shuffle=True, drop_last=False, transform=transform_train, device=self.device)
        test_dataset = GPUDataLoader(
            test_dataset, batch_size=1024, shuffle=True, device=self.device, transform=transform_test)
        # Make n passes to extract the features
        for _ in range(iterations):
            for data, target in train_dataset:
                features_train.append(resnet18(data))
                target_train.append(target)
        for data, target in test_dataset:
            features_test.append(resnet18(data))
            target_test.append(target)

        # Concatenate the features
        features_train = cat(features_train)
        target_train = cat(target_train)
        features_test = cat(features_test)
        target_test = cat(target_test)
        # Save the features
        save(features_train,
             f"{folder}/{task}_{iterations}_features_train.pt")
        save(
            target_train, f"{folder}/{task}_{iterations}_target_train.pt")
        save(features_test,
             f"{folder}/{task}_{iterations}_features_test.pt")
        save(
            target_test, f"{folder}/{task}_{iterations}_target_test.pt")

    def to_dataset(self, train_x, train_y, test_x, test_y):
        """ Create a DataLoader to load the data in batches

        Args:
            train_x (tensor): Training data
            train_y (tensor): Training labels
            test_x (tensor): Testing data
            test_y (tensor): Testing labels
            batch_size (int): Batch size

        Returns:
            DataLoader, DataLoader: Training and testing DataLoader

        """
        train_dataset = GPUTensorDataset(
            train_x, Tensor(train_y).type(
                LongTensor), device=self.device)
        test_dataset = GPUTensorDataset(test_x, Tensor(test_y).type(
            LongTensor), device=self.device)
        return train_dataset, test_dataset

    def normalization(self, train_x, test_x):
        """ Normalize the pixels in train_x and test_x using transform

        Args:
            train_x (np.array): Training data
            test_x (np.array): Testing data

        Returns:
            tensor, tensor: Normalized training and testing data
        """
        # Completely convert train_x and test_x to float torch tensors
        train_x = from_numpy(train_x).float() / 255
        test_x = from_numpy(test_x).float() / 255
        if len(train_x.size()) == 3:
            train_x = train_x.unsqueeze(1)
            test_x = test_x.unsqueeze(1)
        transform = v2.Compose([
            # compute mean and std on shape except channels
            v2.Normalize(mean=train_x.mean(dim=(0, 2, 3)),
                         std=train_x.std(dim=(0, 2, 3))),
            v2.Pad(self.padding, fill=0, padding_mode='constant'),
        ])
        train_x, test_x = transform(train_x), transform(test_x)
        return train_x, test_x

    def read_cifar100(self):
        with open(PATH_CIFAR100_DATABATCH[0], "rb") as f:
            data = pickle.load(f, encoding="bytes")
            training_data = data[b"data"]
            fine_labels = data[b"fine_labels"]
            coarse_labels = data[b"coarse_labels"]
        with open(PATH_CIFAR100_TESTBATCH, "rb") as f:
            data = pickle.load(f, encoding="bytes")
            test_data = data[b"data"]
            test_fine_labels = data[b"fine_labels"]
            test_coarse_labels = data[b"coarse_labels"]
        return training_data, fine_labels, test_data, test_fine_labels, coarse_labels, test_coarse_labels

    def domain_incremental_cifar100(
        self, feature_extraction=False, full=False, *args, **kwargs
    ):
        if not os.path.exists("datasets/CIFAR100/raw"):
            datasets.CIFAR100("datasets", download=True)
        train_datasets, test_datasets = self.cifar100_cil_dataset_generation(
            full=full)
        if feature_extraction:
            resnet = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT
            )
            features = Sequential(
                *list(resnet.children())[:-1])
            transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
            for i in range(len(train_datasets)):
                train_dataset = train_datasets[i]
                test_dataset = test_datasets[i]
                train_datasets[i] = self.set_to_feature_set(
                    train_dataset, features, transform)
                test_datasets[i] = self.set_to_feature_set(
                    test_dataset, features, transform)
        return train_datasets, test_datasets

    def cifar100_cil_dataset_generation(self, full=False):
        # Normalize and pad the data
        training_data, fine_labels, test_data, test_fine_labels, coarse_labels, test_coarse_labels = self.read_cifar100()
        training_data = training_data.reshape(-1, 3, 32, 32)
        test_data = test_data.reshape(-1, 3, 32, 32)
        training_data, test_data = self.normalization(training_data, test_data)
        rescale = v2.Resize(224)
        training_data = rescale(training_data)
        test_data = rescale(test_data)
        # scale data to imagenet size
        # I want to retrieve the class number for each fine label, and sort them by coarse label
        fine_to_coarse = {}
        for (fine, coarse) in zip(fine_labels, coarse_labels):
            if fine not in fine_to_coarse:
                fine_to_coarse[fine] = coarse
        fine_to_coarse = dict(sorted(fine_to_coarse.items()))
        # Organize fine labels by coarse labels (superclasses)
        coarse_to_fine = defaultdict(list)
        for fine, coarse in fine_to_coarse.items():
            coarse_to_fine[coarse].append(fine)
        coarse_to_fine = dict(sorted(coarse_to_fine.items()))
        selected_classes = set()
        datasets_class_mapping = []
        for i in range(5):
            dataset_fine_classes = []
            for coarse, fine_list in coarse_to_fine.items():
                available_fine_classes = [
                    fine for fine in fine_list if fine not in selected_classes]
                if available_fine_classes:
                    chosen_fine = np.random.choice(available_fine_classes)
                    dataset_fine_classes.append(chosen_fine)
                    selected_classes.add(chosen_fine)
            datasets_class_mapping.append(dataset_fine_classes)
        train_datasets = []
        test_datasets = []
        for i, dataset_fine_classes in enumerate(datasets_class_mapping):
            train_x, train_y = [], []
            test_x, test_y = [], []
            # training data
            for j, fine in enumerate(fine_labels):
                if fine in dataset_fine_classes:
                    train_x.append(training_data[j])
                    train_y.append(coarse_labels[j])
            train_x = from_numpy(np.array(train_x).reshape(-1, 3, 224, 224))
            train_y = from_numpy(np.array(train_y))
            # testing data
            for j, fine in enumerate(test_fine_labels):
                if fine in dataset_fine_classes:
                    test_x.append(test_data[j])
                    test_y.append(test_coarse_labels[j])
            test_x = from_numpy(np.array(test_x).reshape(-1, 3, 224, 224))
            test_y = from_numpy(np.array(test_y))
            # normalize and pad the data
            train_dataset, test_dataset = self.to_dataset(
                train_x, train_y, test_x, test_y)
            # extract the features from each dataset
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)
        if full:
            # blend all the datasets
            train_x, train_y = [], []
            test_x, test_y = [], []
            for i in range(len(train_datasets)):
                train_x.append(train_datasets[i].data)
                train_y.append(train_datasets[i].targets)
                test_x.append(test_datasets[i].data)
                test_y.append(test_datasets[i].targets)
            train_x = cat(train_x)
            train_y = cat(train_y)
            test_x = cat(test_x)
            test_y = cat(test_y)
            train_datasets = [GPUTensorDataset(
                train_x, train_y, device=self.device)]
            test_datasets = [GPUTensorDataset(
                test_x, test_y, device=self.device)]
        return train_datasets, test_datasets

    def set_to_feature_set(self, dataset, features, transform):
        batch_size = 64
        number_of_batches = len(dataset) // batch_size if len(
            dataset) % batch_size == 0 else len(dataset) // batch_size + 1
        storage = []
        for i in range(number_of_batches):
            with no_grad():
                batch, targets = dataset[i * batch_size:(i + 1) * batch_size]
                batch = transform(batch)
                batch = features(batch)
                storage.append((batch.to("cpu"), targets.to("cpu")))
        # turn storage into a dataset
        features, targets = zip(*storage)
        features, targets = cat(features), cat(targets)
        return GPUTensorDataset(features, targets, device=self.device)

    def core50(self, scenario="ni", run=0, download=True, *args, **kwargs):
        return CORe50(scenario=scenario, run=run, download=download,
                      device=self.device).get_dataset()


class CORe50:
    """ Load the CORe50 dataset
    INSPIRED BY Vincenzo Lomonaco

    Args:
        root (str, optional): Root folder for the dataset. Defaults to "datasets".
        scenario (str, optional): Scenario to load. Defaults to "ni".
        run (int, optional): Run to load. Defaults to 0.
        start_batch (int, optional): Starting batch. Defaults to 0.
        download (bool, optional): Download the dataset. Defaults to True.
        device (str, optional): Device to use. Defaults to "cuda:0".
    """

    def __init__(self, root="datasets", scenario="ni", run=0, download=True, device="cuda:0"):
        self.root = os.path.join(root, "core50")
        self.scenario = scenario
        self.run = run
        self.device = device
        self.batch_scenario = {
            "ni": 8,
            'nc': 9,
            'nic': 79,
            'nicv2_79': 79,
            'nicv2_196': 196,
            'nicv2_391': 391
        }
        self.md5 = {
            "core50_imgs.npz": "3689d65d0a1c760b87821b114c8c4c6c",
            "labels.pkl": "281c95774306a2196f4505f22fd60ab1",
            "paths.pkl": "b568f86998849184df3ec3465290f1b0",
            "LUP.pkl": "33afc26faa460aca98739137fdfa606e"
        }
        if not os.path.exists(self.root) or not os.listdir(self.root):
            os.makedirs(self.root, exist_ok=True)
            self.download_dataset()

        bin_path = os.path.join(self.root, "core50_imgs.bin")
        if not os.path.exists(bin_path):
            data = np.load(os.path.join(self.root, "core50_imgs.npz"))['x']
            data.tofile(bin_path)

        self.data = np.fromfile(bin_path, dtype=np.uint8).reshape(
            164866, 128, 128, 3)
        self.labels = pickle.load(
            open(os.path.join(self.root, "labels.pkl"), "rb"))
        self.paths = pickle.load(
            open(os.path.join(self.root, "paths.pkl"), "rb"))
        self.lup = pickle.load(open(os.path.join(self.root, "LUP.pkl"), "rb"))

    def download_dataset(self):
        """ Download the dataset """
        files_to_download = [
            ("core50_imgs.npz", REPOSITORY_CORE50_NPZ_128),
            ("paths.pkl", REPOSITORY_CORE50_PATHS),
            ("labels.pkl", REPOSITORY_CORE50_LABELS),
            ("LUP.pkl", REPOSITORY_CORE50_LUP)
        ]
        for file_name, url in files_to_download:
            file_path = os.path.join(self.root, file_name)
            if not os.path.exists(file_path):
                print(f"Downloading {file_name}...")
                self.download_file(url, file_path)

    def checksum(self, file_path):
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5()
            while chunk := f.read(4096):
                file_hash.update(chunk)
        return file_hash.hexdigest()

    def download_file(self, url, file_path):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes,
                            unit='iB', unit_scale=True)
        with open(file_path, 'wb') as file:
            for data in response.iter_content(1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if not self.checksum(file_path) == self.md5[os.path.basename(file_path)]:
            print("Checksum failed. Deleting file.")
            os.remove(file_path)
            sys.exit(1)
        else:
            print("Checksum validated for " + file_path)

    def get_dataset(self):
        """ Returns the train and test sequential datasets"""

        test_indexes = self.lup[self.scenario][self.run][-1]
        test_x = tensor(self.data[test_indexes]).float().to("cpu")
        test_x = test_x.permute(0, 3, 1, 2) / 255
        v2.Normalize((0,), (1,), inplace=True)(test_x, test_x)
        test_y = tensor(
            self.labels[self.scenario][self.run][-1]).to("cpu")
        test_dataset = GPUTensorDataset(test_x, test_y, device=self.device)
        train_loader = []
        for i in range(self.batch_scenario[self.scenario]):
            train_indexes = self.lup[self.scenario][self.run][i]
            train_x = tensor(self.data[train_indexes]).float().to("cpu")
            train_x = train_x.permute(0, 3, 1, 2) / 255
            train_y = tensor(
                self.labels[self.scenario][self.run][i]).to("cpu")
            train_loader.append(
                GPUTensorDataset(train_x, train_y, device=self.device))
        return train_loader, test_dataset
