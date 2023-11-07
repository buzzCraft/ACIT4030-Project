from src.pointnet.utils import PointCloudData, get_transforms
from src.pointnet.pointnet import pointnetloss
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
import os

def set_seed(seed_value=42):
    random.seed(seed_value)       # Python random module
    np.random.seed(seed_value)    # Numpy module
    torch.manual_seed(seed_value) # PyTorch
    if torch.cuda.is_available(): # If running on GPU
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def dataload(path):
    set_seed(42)
    # Load dataset
    train_transforms = get_transforms()
    train_ds = PointCloudData(path, transform=train_transforms)
    valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms)
    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    print('Train dataset size: ', len(train_ds))
    print('Valid dataset size: ', len(valid_ds))
    print('Number of classes: ', len(train_ds.classes))
    print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())
    print('Class: ', inv_classes[train_ds[0]['category']])
    classes = valid_ds.classes
    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=64)
    return train_loader, valid_loader, classes

def train(model, train_loader, optimizer, val_loader=None,  epochs=15, save=True, save_path='model_name/'):
    # Set device to GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = model(inputs.transpose(1,2))

            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                    print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                        (epoch + 1, i + 1, len(train_loader), running_loss / 10))
                    running_loss = 0.0

        model.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = model(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            print('Valid accuracy: %d %%' % val_acc)

        # save the model
        if save:
            if not os.path.isdir(save_path):
                # If not, create the directory
                os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), str(save_path) +str(epoch)+".pth")