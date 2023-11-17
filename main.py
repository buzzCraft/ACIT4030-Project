from pathlib import Path
from src.pointnet.pointnet import PointNet
from src.pointnetPlusPlus.pointnetPlusPlus import pointnetPlusPlus
import src.pointnetTrainer
import src.pointnetPlusPlusTrainer
import torch
from src.utils.data_utils import ModelNetDataLoader

from src.benchmark import tests


def __main__():
    # Set device to GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set path to dataset
    path = Path("data\ModelNet10")

    # Set epochs
    epochs = 10

    # Load dataset without farthest point sampling
    train_dataset = ModelNetDataLoader(path, split="train", use_uniform_sample=True)
    trainDataLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=24,
        shuffle=True,
        drop_last=True,
    )

    # Load pointnet model
    pointnet = PointNet().to(device)

    # Train pointnet
    src.pointnetTrainer.train(
        model=pointnet,
        train_loader=trainDataLoader,
        save_path="models/pointnet",
        epochs=epochs,
    )

    # Load dataset with farthest point sampling
    train_dataset = ModelNetDataLoader(path, split="train", use_uniform_sample=True)
    trainDataLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=24,
        shuffle=True,
        drop_last=True,
    )

    # Load pointnet++ model
    pointnetPlusPlusModel = pointnetPlusPlus()

    # Train pointnet++
    src.pointnetPlusPlusTrainer.train(
        model=pointnetPlusPlusModel,
        train_loader=trainDataLoader,
        save_path="models/pointnetplusplus",
        epochs=epochs,
    )

    # Set up test datasets for testing
    valid_dataset_pointnet = ModelNetDataLoader(
        path, split="test", use_uniform_sample=False
    )
    valid_dataset_pointnetPlusPlus = ModelNetDataLoader(
        path, split="test", use_uniform_sample=True
    )

    # Load models for testing
    pointnet = PointNet().to(device)
    pointnetPlusPlusModel = pointnetPlusPlus().to(device)

    # test trained models
    tests.test_models(
        models={"PointNet": pointnet, "PointNet++": pointnetPlusPlusModel},
        model_paths={
            "PointNet": f"models/pointnet/pointnet_epochs{epochs}.pth",
            "PointNet++": f"models/pointnetplusplus/pointnetplusplus_epochs{epochs}.pth",
        },
        valid_datasets=[valid_dataset_pointnet, valid_dataset_pointnetPlusPlus],
    )


if __name__ == "__main__":
    __main__()
