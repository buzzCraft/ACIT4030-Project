"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import pointnetPlusPlus.provider as provider
import importlib


from pathlib import Path
from tqdm import tqdm
from pointnetPlusPlus.data_utils import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "pointnetPlusPlus"))

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


def test(model, loader, num_class=40, use_cpu=False):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

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

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


def train(use_cpu=False, gpu="0", batch_size=24, model="models", num_category=10, epoch=200, learning_rate=0.001, num_point=1024, optimizer="Adam", decay_rate=1e-4, use_normals=False, process_data=False, use_uniform_sample=False):

    """HYPER PARAMETER"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    """DATA LOADING"""
    data_path = "data/modelnet40_normal_resampled/"

    train_dataset = ModelNetDataLoader(
        root=data_path, num_point=num_point, split="train", use_uniform_sample=use_uniform_sample, use_normals=use_normals, process_data=process_data, num_category=num_category
    )
    test_dataset = ModelNetDataLoader(
        root=data_path, num_point=num_point, split="test", use_uniform_sample=use_uniform_sample, use_normals=use_normals, process_data=process_data, num_category=num_category
    )
    trainDataLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=10,
        drop_last=True,
    )
    testDataLoader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=10
    )

    """MODEL LOADING"""
    num_class = num_category
    model = importlib.import_module(model)


    classifier = model.get_model(num_class, normal_channel=use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    start_epoch = 0

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
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    """TRANING"""
    for epoch in range(start_epoch, epoch):
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, (points, target) in tqdm(
            enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9
        ):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            # Inside your training loop
            running_loss += loss.item()
            if batch_id % 10 == 9:    # print every 10 mini-batches
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                    (epoch + 1, batch_id + 1, len(trainDataLoader), running_loss / 10))
                running_loss = 0.0
            global_step += 1

        train_instance_acc = np.mean(mean_correct)

        with torch.no_grad():
            instance_acc, class_acc = test(
                classifier.eval(), testDataLoader, num_class=num_class, use_cpu=use_cpu,
            )

            if instance_acc >= best_instance_acc:
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if class_acc >= best_class_acc:
                best_class_acc = class_acc

            global_epoch += 1
