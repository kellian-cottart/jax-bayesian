from torch.utils.data import Dataset
from torch import tensor, isin, unique, randperm


class GPUTensorDataset(Dataset):
    """ Dataset which has a data and a targets tensor, designed to be used on GPU

    Args:
        data (torch.tensor): Data tensor
        targets (torch.tensor): Targets tensor
        device (str, optional): Device to use. Defaults to "cuda:0".
        transform (torchvision.transforms.Compose, optional): Data augmentation transform. Defaults to None.
    """

    def __init__(self, data, targets, device="cuda:0"):
        self.data = data.to("cuda:0")
        self.targets = targets.to("cuda:0")
        self.device = device

    def __getitem__(self, index):
        """ Return a (data, target) pair """
        return self.data[index].to(self.device), self.targets[index].to(self.device)

    def __len__(self):
        """ Return the number of samples """
        return len(self.data)

    def shuffle(self):
        """ Shuffle the data and targets tensors """
        perm = randperm(len(self.data), device="cuda:0")
        self.data = self.data[perm]
        self.targets = self.targets[perm]

    def __getbatch__(self, start, batch_size):
        """ Return a batch of data and targets """
        if start+batch_size > len(self.data):
            return self.data[start:], self.targets[start:]
        if batch_size == 1:
            return self.data[start].unsqueeze(0), self.targets[start].unsqueeze(0)
        return self.data[start:start+batch_size], self.targets[start:start+batch_size]

    def __getclasses__(self, class_indexes):
        """ Return a GPUTensorDataset with only the specified classes """
        class_indexes = tensor(class_indexes, device="cuda:0")
        indexes = isin(self.targets, class_indexes)
        return GPUTensorDataset(self.data[indexes], self.targets[indexes], device=self.device)

    def __repr__(self):
        return f"GPUTensorDataset(data={self.data.shape},targets={self.targets.shape}, classes={unique(self.targets)} ,device_idle={self.data.device}, device_train={self.device})"
