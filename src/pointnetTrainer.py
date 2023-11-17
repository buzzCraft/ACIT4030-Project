from src.pointnet.pointnet import pointnetloss
import numpy as np
import torch
import os
from tqdm import tqdm
import src.utils.transformations as transformations


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
        mean_correct = []

        scheduler.step()

        tqdm_loader = tqdm(enumerate(train_loader, 0), total=len(train_loader))

        for i, (points, target) in tqdm_loader:
            points = points.data.numpy()
            points = transformations.random_point_dropout(points)
            points[:, :, 0:3] = transformations.random_scale_point_cloud(
                points[:, :, 0:3]
            )
            points[:, :, 0:3] = transformations.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            points, target = points.to(device), target["category"].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = model(points)

            pred_choice = outputs.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

            loss = pointnetloss(outputs, target, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                message = "[Epoch: %d], loss: %.3f, training accuracy: %.3f" % (
                    epoch + 1,
                    running_loss / 10,
                    np.mean(mean_correct),
                )
                tqdm_loader.set_description(message)
                running_loss = 0.0

        model.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                val_acc = []
                tqdm_loader = tqdm(enumerate(val_loader, 0), total=len(val_loader))

                for i, (points, target) in tqdm(tqdm_loader):
                    points, target = points.to(device).float(), target["category"].to(
                        device
                    )
                    points = torch.Tensor(points)
                    points = points.transpose(2, 1)
                    outputs, __, __ = model(points)
                    pred_choice = outputs.data.max(1)[1]

                    correct = pred_choice.eq(target.long().data).cpu().sum()
                    val_acc.append(correct.item() / float(points.size()[0]))
            tqdm_loader.write("Valid accuracy: %d %%" % np.mean(val_acc))

        # save the model
        if save_path:
            if not os.path.isdir(save_path):
                # If not, create the directory
                os.makedirs(save_path, exist_ok=True)
            filepath = os.path.join(save_path, f"pointnet_epochs{epoch+1}.pth")
            torch.save(model.state_dict(), filepath)
