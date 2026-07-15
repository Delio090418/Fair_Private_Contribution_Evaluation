import torch
import numpy as np
import torch.nn as nn
import math
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import random_split,DataLoader, Subset, TensorDataset,Dataset
import random
from collections import defaultdict
import pathlib
import matplotlib.pyplot as plt


datamnist='/Users/delio/Documents/Working_projects/Balazs/Experiments/MNIST/data_mnist'
#transform and path for brain data set
datacifar10="/Users/delio/Documents/Current_working_projects/data"

def load_data_set(name="cifar10"):
    if name == "cifar10":
        dataset=datasets.CIFAR10(root=datacifar10, train=True, download=False, transform=ToTensor())
    elif name == "mnist":
        dataset=datasets.MNIST(root=datamnist, train=True, download=False, transform=ToTensor())
    else:
        raise ValueError("Not support dataset")
    return dataset

def load_test_set(name="cifar10"):
    if name == "cifar10":
        dataset=datasets.CIFAR10(root=datacifar10, train=False, download=False, transform=ToTensor())
    elif name == "mnist":
        dataset=datasets.MNIST(root=datamnist, train=False, download=False, transform=ToTensor())
    else:
        raise ValueError("Not support dataset")
    return dataset
    
def equal_label_partition(dataname, num_clients=3, seed=0):
    """
    Split dataset equally among clients with equal samples per label (IID).

    Args:
        dataset: torch.utils.data.Dataset (MNIST or CIFAR10)
        num_clients: int, number of clients
        seed: int, random seed

    Returns:
        client_indices: dict {client_id: [sample_indices]}
        label_distribution: dict {client_id: {label: count}}
    """

    dataset =  load_data_set(dataname)

    rng = np.random.default_rng(seed)
    
    # Extract labels
    labels = np.array(dataset.targets if hasattr(dataset, 'targets') else dataset.labels)
    num_classes = len(np.unique(labels))
    
    # Indices for each class
    indices_per_class = {c: np.where(labels == c)[0] for c in range(num_classes)}
    for c in range(num_classes):
        rng.shuffle(indices_per_class[c])
    
    # Equal per-label split
    client_indices = defaultdict(list)
    label_distribution = defaultdict(lambda: defaultdict(int))

    for c in range(num_classes):
        class_indices = indices_per_class[c]
        n_per_client = len(class_indices) // num_clients
        
        # Split each class evenly among clients
        splits = np.array_split(class_indices[:n_per_client * num_clients], num_clients)
        for i, split in enumerate(splits):
            client_indices[i].extend(split)
            label_distribution[i][c] = len(split)
    
    return client_indices, label_distribution


def create_client_datasets(full_dataset, client_indices_dict):
    """
    full_dataset: any PyTorch dataset (e.g. CIFAR-10)
    client_indices_dict: {client_id: [list of sample indices]}
    
    Returns:
        {client_id: Subset(dataset, indices)}
    """
    client_datasets = {}
    
    for client_id, indices in client_indices_dict.items():
        client_datasets[client_id] = Subset(full_dataset, indices)
    
    return client_datasets

class NoisyClientDataset(Dataset):
    def __init__(self, base_dataset, indices, num_classes=10, noise_rate=0.0):
        """
        base_dataset: original CIFAR-10 dataset
        indices: list of indices belonging to this client
        num_classes: number of classes (CIFAR-10 = 10)
        noise_rate: fraction of labels to corrupt for this client
        """
        self.base_dataset = base_dataset
        self.indices = indices
        self.num_classes = num_classes
        
        # Extract original labels for this client
        self.labels = torch.tensor([base_dataset[i][1] for i in indices])
        
        # Apply noise
        self.noisy_labels = self._add_symmetric_noise(self.labels, noise_rate)

    def _add_symmetric_noise(self, labels, noise_rate):
        noisy = labels.clone()
        n = len(labels)
        
        # Choose positions to corrupt
        mask = torch.rand(n) < noise_rate
        
        # Random new labels DIFFERENT from true label
        random_labels = torch.randint(0, self.num_classes, (mask.sum(),))
        conflicts = random_labels == noisy[mask]
        
        # Fix cases where random_labels == true labels
        while conflicts.any():
            random_labels[conflicts] = torch.randint(0, self.num_classes, (conflicts.sum(),))
            conflicts = random_labels == noisy[mask]
        
        noisy[mask] = random_labels
        return noisy
    


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, _ = self.base_dataset[real_idx]
        return img, self.noisy_labels[idx].item()
    

def create_noisy_clients(name, client_indices_dict, noise_rates_dict):
    """
    base_dataset: CIFAR10 dataset
    client_indices_dict: {client_id: [list of indices]}
    noise_rates_dict: {client_id: noise_rate}

    Returns:
        {client_id: NoisyClientDataset}
    """
    base_dataset = load_data_set(name=name)
    clients = {}

    for client_id, indices in client_indices_dict.items():
        noise = noise_rates_dict.get(client_id, 0.0)
        
        clients[client_id] = NoisyClientDataset(
            base_dataset=base_dataset,
            indices=indices,
            num_classes=10,
            noise_rate=noise
        )
    
    return clients

def cl_data_load_and_testset(dataname="cifar10", num_clients=3, noise_clients={0:0.0,1:0.1,2:0.2}):

    test_set = load_test_set(name=dataname)

    test_loader = DataLoader(test_set,batch_size=128,shuffle=False)

    client_data = {}


# Split equally per label among 10 clients
    clients, label_dist = equal_label_partition(dataname=dataname, num_clients=num_clients, seed=0)

    clients_data = create_noisy_clients(name=dataname, client_indices_dict=clients, noise_rates_dict=noise_clients)

    for idx, client in clients_data.items():
        
        client_data[idx] = {
            'train_loader': DataLoader(client,batch_size=64,shuffle=True)
        }
    
    return client_data, test_loader




if __name__ == "__main__":

    data_name = "cifar10"
    noise_dict = {0:0.0,1:0.2,2:0.5}
    # clients_data = equal_label_partition(dataname=data_name, num_clients=3)
    # noise_clients = create_noisy_clients(data_name,clients_data[0], noise_dict)
    noise_dataloader = cl_data_load_and_testset(data_name,3, noise_dict)
    # print(len(clients_data[0][0]))
    # print(len(noise_clients[0]))
    print(len(noise_dataloader[0]["train_loader"].dataset))