from src.pointnet.utils import PointCloudData, get_transforms
from src.pointnet.pointnet import pointnetloss
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
import os
from tqdm import tqdm


def train(
    model,
    train_loader,
    val_loader=None,
    epochs=15,
    save_path=None,
    learning_rate=0.001,
    decay_rate=0.0001,
    optimizer="Adam",
):
    # Set device to GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Optimizer setup
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        scheduler.step()

        tqdm_loader = tqdm(enumerate(train_loader, 0), total=len(train_loader))

        for i, (points, target) in tqdm_loader:
            points = points.data.numpy()
            # points = provider.random_point_dropout(points)
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            points, target = points.to(device), target["category"].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = model(points)

            loss = pointnetloss(outputs, target, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                message = "[Epoch: %d, Batch: %4d / %4d], loss: %.3f" % (
                    epoch + 1,
                    i + 1,
                    len(train_loader),
                    running_loss / 10,
                )
                tqdm_loader.set_description(message)
                running_loss = 0.0

        model.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                for i, (points, target) in tqdm(
                    enumerate(train_loader, 0), total=len(train_loader)
                ):
                    points, target = points.to(device).float(), target["category"].to(
                        device
                    )
                    points = torch.Tensor(points)
                    points = points.transpose(2, 1)
                    outputs, __, __ = model(points)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            val_acc = 100.0 * correct / total
            print("Valid accuracy: %d %%" % val_acc)

        # save the model
        if save_path:
            save_path = os.path.join(save_path, f'pointnet_epochs{epoch}.pth')
            if not os.path.isdir(save_path):
                # If not, create the directory
                os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), save_path)
