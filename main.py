from pathlib import Path
import os
import numpy as np
from src.pointnet.utils import read_file
from src.visualization import show_mesh, show_scatter
import src.pointnetTrainer
from src.pointnet.pointnet import PointNet
import src.pointnetPlusPlusTrainer

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = Path('data\ModelNet10')
# # Look at the data
# with open(path / "bed/train/bed_0002.off", 'r') as f:
#     verts, faces = read_file(f)
#
# i,j,k = np.array(faces).T
# x,y,z = np.array(verts).T
#
# show_mesh(x,y,z,i,j,k)
# show_scatter(x,y,z)

pointnet = PointNet()
pointnet.to(device)
optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.001)
train_loader, test_loader, _ = src.utils.dataload(path)
src.pointnetTrainer.train(model = pointnet, train_loader=train_loader, optimizer = optimizer, save_path='pointnetmodel/pointnet')
src.pointnetPlusPlusTrainer.train()