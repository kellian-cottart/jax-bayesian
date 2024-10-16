from .structures import *
from torch import cat


def permuted_dataset(dataset, batch_size, continual, task_id, iteration, max_iterations, permutations, epoch):
    """Permute a given batch of a dataset and return the permuted data and targets.

        dataset (object): The dataset object.
        batch_size (int): The size of each batch.
        continual (bool): Whether to use continual learning.
        task_id (int): The ID of the current task.
        iteration (int): The current iteration.
        max_iterations (int): The maximum number of iterations.
        permutations (list): A list of permutations for each task.
        epoch (int): The current epoch.

        tuple: A tuple containing the permuted batch data and the targets.
    """
    perm = permutations[task_id]
    batch_data, targets = dataset.__getbatch__(
        batch_size * iteration, batch_size)
    shape = batch_data.shape
    batch_data = batch_data.to(perm.device).view(
        shape[0], shape[1], -1)
    split = 0.75
    n_images_taken_b1 = int((1 - (iteration*(epoch+1) - int(max_iterations*split)) / (
        max_iterations - int(max_iterations*split))) * batch_size)  # Ratio between the number of images taken from batch perm 1 and batch perm 2
    if len(permutations) > task_id + 1 and batch_size - n_images_taken_b1 > 0 and continual == True:
        next_perm = permutations[task_id + 1]
        batch_data = cat(
            (batch_data[:n_images_taken_b1, :, perm],
                batch_data[n_images_taken_b1:, :, next_perm]), dim=0).view(shape)
    else:
        batch_data = batch_data[:, :, perm].view(shape)
    return batch_data, targets


def permuted_labels(dataset, batch_size, task_id, permutations, iteration):
    """Permute the labels of a given batch of a dataset and return the permuted data and targets.
    """
    permutation = permutations[task_id]
    batch_data, targets = dataset.__getbatch__(
        batch_size * iteration, batch_size)
    targets = targets.to(permutation.device)
    # the list of permutation give us the new labels for each class, we need to map the old labels to the new ones
    targets = permutation[targets]
    return batch_data, targets


def batch_yielder(dataset, task, batch_size=128, continual=None, task_id=None, iteration=None, max_iterations=None, permutations=None, epoch=None):
    batch_data, targets = None, None

    if "PermutedLabels" in task:
        batch_data, targets = permuted_labels(
            dataset=dataset, batch_size=batch_size, task_id=task_id, permutations=permutations, iteration=iteration)
    elif "Permuted" in task:
        batch_data, targets = permuted_dataset(dataset=dataset, batch_size=batch_size, continual=continual,
                                               task_id=task_id, iteration=iteration, max_iterations=max_iterations, permutations=permutations, epoch=epoch)
    else:
        batch_data, targets = dataset.__getbatch__(
            batch_size * iteration, batch_size)
    return batch_data.to(dataset.device), targets.to(dataset.device)


def test_permuted_dataset(test_dataset, permutations):
    for i in range(len(permutations)):
        data, targets = permuted_dataset(dataset=test_dataset, batch_size=test_dataset.data.shape[
                                         0], continual=False, task_id=i, iteration=0, max_iterations=1, permutations=permutations, epoch=0)
        yield GPUTensorDataset(data, targets, device=test_dataset.device)


def test_permuted_labels(test_dataset, permutations):
    for i in range(len(permutations)):
        data, targets = permuted_labels(
            dataset=test_dataset, batch_size=test_dataset.data.shape[0], task_id=i, permutations=permutations, iteration=0)
        yield GPUTensorDataset(data, targets, device=test_dataset.device)
