from pathlib import Path
import os
import numpy as np
from src.pointnet.utils import read_file
from src.visualization import show_mesh, show_scatter
import src.pointnetTrainer
from src.pointnet.pointnet import PointNet
from src.pointnetPlusPlus.pointnetPlusPlus import pointnetPlusPlus
import src.pointnetPlusPlusTrainer
import torch
from src.data_utils import ModelNetDataLoader


def __main__():
    # Set device to GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set path to dataset
    path = Path("data\ModelNet10")

    # Load dataset
    train_dataset = ModelNetDataLoader(path, split="train", use_uniform_sample=False)
    trainDataLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=24,
        shuffle=True,
        drop_last=True,
    )

    # Load model and optimizer for pointnet
    pointnet = PointNet().to(device)

    # # Train pointnet
    src.pointnetTrainer.train(
        model=pointnet,
        train_loader=trainDataLoader,
        save_path="models/pointnet",
        epochs=1,
    )

    # Load dataset with farthest point sampling
    train_dataset = ModelNetDataLoader(path, split="train", use_uniform_sample=False)
    trainDataLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=24,
        shuffle=True,
        drop_last=True,
    )

    pointnetPlusPlusModel = pointnetPlusPlus()
    src.pointnetPlusPlusTrainer.train(
        model=pointnetPlusPlusModel,
        train_loader=trainDataLoader,
        save_path="models/pointnetplusplus",
        epochs=1,
    )


if __name__ == "__main__":
    __main__()
