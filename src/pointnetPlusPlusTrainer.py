import os
import sys
import torch
import numpy as np

import src.utils.transformations as transformations
from src.pointnetPlusPlus.pointnetPlusPlus import get_loss

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inplace_relu(m):
    """
    Modify ReLU modules in the model to use inplace operations,
    which can reduce memory usage.
    Input: PyTorch model
    """
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


def test(model, loader, num_class=10):
    """
    Test the model using the provided data loader.
    Calculates the instance and class-wise accuracy.
    Input: model (PyTorch model), loader (data loader), num_class (int)
    Output: instance_acc (float), class_acc (float)
    """
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    tqdm_loader = tqdm(enumerate(loader, 0), total=len(loader))

    # Iterate over the dataset
    for j, (points, target) in tqdm_loader:
        points, target = points.to(device), target["category"].to(device)

        # Preprocess and feed data to the model
        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        # Calculate accuracy
        for cat in np.unique(target.cpu()):
            classacc = (
                pred_choice[target == cat]
                .eq(target[target == cat].long().data)
                .cpu()
                .sum()
            )
            class_acc[cat, 0] += classacc.item() / float(
                points[target == cat].size()[0]
            )
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    # Calculate overall accuracies
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


def train(
    model,
    train_loader,
    val_loader=None,
    epochs=15,
    save_path=None,
    learning_rate=0.001,
    optimizer="Adam",
    decay_rate=1e-4,
):
    """
    Train a model on point cloud data.
    Input: Various hyperparameters and settings
    Output: Trained model
    """
    # Model loading
    model = model

    classifier = model.to(device)
    criterion = get_loss().to(device)
    classifier.apply(inplace_relu)

    start_epoch = 0

    # Optimizer setup
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    for epoch in range(start_epoch, epochs):
        mean_correct = []
        classifier = classifier.train()
        running_loss = 0

        scheduler.step()

        tqdm_loader = tqdm(enumerate(train_loader, 0), total=len(train_loader))

        # Iterating over the dataset
        for batch_id, (points, target) in tqdm_loader:
            optimizer.zero_grad()

            # Preprocess points and feed to the classifier
            points = points.data.numpy()
            points = transformations.random_point_dropout(points)
            points[:, :, 0:3] = transformations.random_scale_point_cloud(
                points[:, :, 0:3]
            )
            points[:, :, 0:3] = transformations.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            points, target = points.to(device), target["category"].to(device)

            # Calculate accuracy and loss
            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_id % 10 == 9:
                tqdm_loader.set_description(
                    "[Epoch: %d], loss: %.3f, training accuracy: %.3f"
                    % (
                        epoch + 1,
                        running_loss / 10,
                        np.mean(mean_correct),
                    )
                )
                running_loss = 0.0

        if val_loader:
            with torch.no_grad():
                instance_acc, class_acc = test(
                    classifier.eval(),
                    val_loader,
                )

                if instance_acc >= best_instance_acc:
                    best_instance_acc = instance_acc
                    best_epoch = epoch + 1

                if class_acc >= best_class_acc:
                    best_class_acc = class_acc

        # save the model
        if save_path:
            if not os.path.isdir(save_path):
                # If not, create the directory
                os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f"pointnetplusplus_epochs{epoch+1}.pth")
            torch.save(model.state_dict(), file_path)
