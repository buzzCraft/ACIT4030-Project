from pathlib import Path
import os
import numpy as np
from src.dataloader import read_file
from src.visualization import show_mesh, show_scatter

path = Path('data\ModelNet10')
with open(path / "bed/train/bed_0001.off", 'r') as f:
    verts, faces = read_file(f)

i,j,k = np.array(faces).T
x,y,z = np.array(verts).T

show_mesh(x,y,z,i,j,k)
show_scatter(x,y,z)