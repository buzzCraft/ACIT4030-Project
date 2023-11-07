from pathlib import Path
import os
import numpy as np
from src.pointnet.utils import read_file
from src.visualization import show_mesh, show_scatter
from src.trainer import dataload, train
from src.pointnet.pointnet import PointNet
from src.pointnext.pointnext import PointNext, PointNextDecoder, pointnext_s
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = Path('data\ModelNet10')
# with open(path / "bed/train/bed_0002.off", 'r') as f:
#     verts, faces = read_file(f)
#
# i,j,k = np.array(faces).T
# x,y,z = np.array(verts).T
#
# show_mesh(x,y,z,i,j,k)
# show_scatter(x,y,z)

# pointnet = PointNet()
# pointnet.to(device)
# optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.001)
# train_loader, test_loader, _ = dataload(path)
# train(model = pointnet, train_loader=train_loader, optimizer = optimizer)


encoder = pointnext_s(in_dim=3)
model = PointNext(40, encoder=encoder).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loader, test_loader, _ = dataload(path)
train(model = model, train_loader=train_loader, optimizer = optimizer)