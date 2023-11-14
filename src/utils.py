import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.pointnet.utils import PointCloudData, get_transforms

# Function to set the seed for reproducibility across various modules
def set_seed(seed_value=42):
    random.seed(seed_value)       # Sets the seed for Python's random module
    np.random.seed(seed_value)    # Sets the seed for NumPy's random number generation
    torch.manual_seed(seed_value) # Sets the seed for PyTorch's random number generation

    # Additional steps for ensuring reproducibility in PyTorch when using a GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)      # Sets the seed for the current GPU
        torch.cuda.manual_seed_all(seed_value)  # Sets the seed for all GPUs
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Function to load datasets
def dataload(path):
    set_seed(42)  # Ensuring reproducibility for dataset loading

    # Load and transform the dataset
    train_transforms = get_transforms()
    train_ds = PointCloudData(path, transform=train_transforms)
    valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms)

    # Inverse mapping from class indices to class names
    inv_classes = {i: cat for cat, i in train_ds.classes.items()}

    # Printing dataset statistics
    print('Train dataset size: ', len(train_ds))
    print('Valid dataset size: ', len(valid_ds))
    print('Number of classes: ', len(train_ds.classes))
    print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())
    print('Class: ', inv_classes[train_ds[0]['category']])

    # Getting classes from the validation dataset
    classes = valid_ds.classes

    # Creating data loaders for training and validation datasets
    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=64)

    # Returning the data loaders and class information
    return train_loader, valid_loader, classes
