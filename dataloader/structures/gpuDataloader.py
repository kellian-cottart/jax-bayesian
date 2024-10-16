from torch import randperm, arange


class GPUDataLoader():
    """ DataLoader which loads a GPUTensorDataset, allowing to load the data on GPU

    Args:
        dataset (GPUTensorDataset): Dataset to load
        batch_size (int): Batch size
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        drop_last (bool, optional): Whether to drop the last batch if it is not full. Defaults to True.
        transform (torchvision.transforms.Compose, optional): Data augmentation transform. Defaults to None.
    """

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True, transform=None, device="cuda:0", test=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.transform = transform
        self.device = device
        self.test = test

    def __iter__(self):
        """ Return an iterator over the dataset """
        self.index = 0
        if self.shuffle:
            self.perm = randperm(len(self.dataset))
        else:
            self.perm = arange(len(self.dataset))
        return self

    def __next__(self):
        """ Return the next batch """
        if self.index >= len(self.dataset):
            raise StopIteration
        if self.index + self.batch_size > len(self.dataset):
            if self.drop_last == False:
                indexes = self.perm[self.index:]
            raise StopIteration
        else:
            indexes = self.perm[self.index:self.index+self.batch_size]
        self.index += self.batch_size
        data, targets = self.dataset.data[indexes], self.dataset.targets[indexes]
        if self.transform is not None:
            data = self.transform(data)
        return data.to(self.device), targets.to(self.device)

    def __len__(self):
        """ Return the number of batches """
        return len(self.dataset)//self.batch_size
