'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''

import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

def pc_normalize(pc):
    # Normalize the point cloud data
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Sample points from point cloud data using farthest point sampling algorithm.
    Input:
        point: point cloud data, [N, D]
        npoint: number of points to sample
    Return:
        Sampled point cloud data, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def read_file(file, num_points=1024, check_num_points=False):
    """
    Reads an OFF (Object File Format) file and extracts the vertices and faces.

    Parameters:
        file : An open file-like object with methods read() or readline()
                                 that return a string.

    Returns:
        tuple: A tuple containing two lists, the first with vertices and the second with faces.
               Vertices are 3-element lists [x, y, z], and faces are lists of indices (as integers)
               corresponding to the vertices that form each face.
    """
    # Check the OFF file format header
    if 'OFF' != file.readline().strip():
        raise ValueError('Not a valid OFF header')

    # Read the number of vertices and faces
    n_verts, _, __ = tuple(int(s) for s in file.readline().strip().split(' '))

    # Check if number of points is correct
    if check_num_points:
        if n_verts < num_points:
            return False
        return True

    # Read the vertices
    verts = [[float(s) for s in file.readline().strip().split(' ')] for _ in range(n_verts)]

    return verts

class ModelNetDataLoader(Dataset):
    def __init__(self, root, num_point=1024, split='train', process_data=False, use_uniform_sample=False, use_normals=True, num_category=10):
        self.root = root
        self.npoints = num_point
        self.process_data = process_data
        self.uniform = use_uniform_sample
        self.use_normals = use_normals
        self.num_category = num_category

        # Gather categories from directory names
        self.cat = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        # Initialize datapath list
        self.datapath = []

        # Iterate through each category
        for category in self.cat:
            dir = os.path.join(self.root, category, split)  # Path to the category directory

            # Check if the directory exists
            if not os.path.exists(dir):
                continue

            # Add file paths to the datapath list
            for filename in os.listdir(dir):
                if filename.endswith('.off'):
                    sample_path = os.path.join(dir, filename)
                    if self._has_enough_points(sample_path):
                        self.datapath.append((category, sample_path))

        print('The size of %s data is %d' % (split, len(self.datapath)))

        # Determine save path based on whether using uniform sampling or not
        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        # Process and save data if not already processed
        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                # Process each data file
                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    # Sample points from point cloud
                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                # Save processed data
                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                # Load processed data if it exists
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def _has_enough_points(self, filepath):
        with open(filepath, 'r') as f:
            enough_points = read_file(f, self.npoints, check_num_points=True)  # Assuming read_file returns vertices and faces
        return enough_points

    def __len__(self):
        # Return the length of the dataset
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)

            # Open and read the OFF file
            with open(fn[1], 'r') as f:
                verts = read_file(f)

            # Convert vertices to a NumPy array
            point_set = np.array(verts, dtype=np.float32)

            # Sample points from point cloud
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        # Normalize point cloud and optionally remove normals
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        # Return the requested item from the dataset
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    # Example usage of the ModelNetDataLoader class
    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        # Print the shape of the points and labels in each batch
        print(point.shape)
        print(label.shape)
