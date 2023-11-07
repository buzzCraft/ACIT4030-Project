from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import numpy as np
import random
from pathlib import Path
import os
import math

# BASED ON https://colab.research.google.com/github/nikitakaraevv/pointnet/blob/master/nbs/PointNetClass.ipynb#scrollTo=zCgPQhfvh7R3

def read_file(file):
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
    n_verts, n_faces, _ = tuple(int(s) for s in file.readline().strip().split(' '))

    # Read the vertices
    verts = [[float(s) for s in file.readline().strip().split(' ')] for _ in range(n_verts)]

    # Read the faces
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for _ in range(n_faces)]

    return verts, faces


def default_transforms():
    """
    Defines and composes transformations that are applied to the point cloud data.

    Returns:
        torchvision.transforms.Compose: A composed transform that includes sampling points,
                                        normalizing, and converting to tensor.
    """
    # Compose the point cloud transformations
    return transforms.Compose([
        PointSampler(1024),  # Sample a fixed number of points
        Normalize(),  # Normalize the point cloud
        ToTensor()  # Convert the point cloud data to a tensor
    ])


class PointSampler(object):
    """
    A transformation that samples a fixed number of points from a given point cloud.
    Provides a constant number of points as input to the model.

    Attributes:
        output_size (int): The number of points to sample from the point cloud.

    Methods:
        triangle_area(pt1, pt2, pt3): Calculates the area of a triangle given by three points.
        sample_point(pt1, pt2, pt3): Samples a random point within a triangle using barycentric coordinates.
        __call__(mesh): Applies the transformation to sample points from the mesh.
    """

    def __init__(self, output_size):
        """
        Initializes the PointSampler transformation.
        Parameters:
            output_size (int): The number of points to sample.
        """
        assert isinstance(output_size, int), "output_size should be an integer."
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        """
        Calculates the area of a triangle specified by three points.
        Parameters:
            pt1, pt2, pt3 (numpy.ndarray): The vertices of the triangle.
        Returns:
            float: The area of the triangle.
        """
        # Calculate the length of the triangle sides
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)

        # Use Heron's formula to calculate the area
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

    def sample_point(self, pt1, pt2, pt3):
        """
        Samples a random point inside the triangle defined by the given vertices.
        This is achieved by using barycentric coordinates to interpolate between the vertices.
        Parameters:
            pt1, pt2, pt3 (numpy.ndarray): The vertices of the triangle.
        Returns:
            tuple: A point (x, y, z) sampled within the triangle.
        """
        # Generate two random numbers for barycentric coordinates
        s, t = sorted([random.random(), random.random()])
        # Interpolate the point inside the triangle
        f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        """
        Applies the sampling transformation to the mesh.
        Parameters:
            mesh (tuple): A tuple containing the vertices and faces of the mesh.
        Returns:
            numpy.ndarray: An array of sampled points from the mesh.
        """
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        # Calculate the area of each triangle face
        for i in range(len(areas)):
            areas[i] = self.triangle_area(verts[faces[i][0]],
                                          verts[faces[i][1]],
                                          verts[faces[i][2]])

        # Sample faces based on their area
        sampled_faces = random.choices(faces, weights=areas, k=self.output_size)

        # Sample points from the selected faces
        sampled_points = np.zeros((self.output_size, 3))
        for i in range(len(sampled_faces)):
            sampled_points[i] = self.sample_point(verts[sampled_faces[i][0]],
                                                  verts[sampled_faces[i][1]],
                                                  verts[sampled_faces[i][2]])

        return sampled_points


class Normalize(object):
    """
    Normalizes a point cloud by centering it around the origin and scaling it to fit
    within a unit sphere.
    """
    def __call__(self, pointcloud):
        """
        Applies normalization to a given point cloud.
        Parameters:
            pointcloud (numpy.ndarray): A 2D array representing a point cloud with
                                        shape (num_points, num_dimensions).
        Returns:
            numpy.ndarray: The normalized point cloud.
        """
        # Verify the shape of the point cloud
        assert len(pointcloud.shape) == 2, "Point cloud must be a 2D array."

        # Center the point cloud at the origin
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        # Scale the point cloud to fit inside a unit sphere
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return norm_pointcloud

class ToTensor(object):
    """
    Converts a numpy.ndarray (N x D) to a torch.FloatTensor of shape (D x N).
    """
    def __call__(self, pointcloud):
        """
        Converts the given point cloud to a PyTorch tensor.
        Parameters:
            pointcloud (numpy.ndarray): A 2D array representing a point cloud with
                                        shape (num_points, num_dimensions).
        Returns:
            torch.FloatTensor: A tensor representation of the point cloud.
        """
        # Verify the shape of the point cloud
        assert len(pointcloud.shape) == 2, "Point cloud must be a 2D array."

        # Convert the numpy array to a PyTorch tensor
        return torch.from_numpy(pointcloud)


class PointCloudData(Dataset):
    """
    A custom PyTorch Dataset for loading and transforming point cloud data from OFF files.
    Attributes:
        root_dir (str or Path): The root directory where the dataset is stored.
        classes (dict): A mapping from folder names to class indices.
        transforms (callable): A function/transform that takes in a sample and returns a transformed version.
        valid (bool): A flag used to determine if the dataset is used for validation.
        files (list): A list of dictionaries, each containing the file path and category of a point cloud.
    Methods:
        __len__(): Returns the number of items in the dataset.
        __preproc__(file): Processes the OFF file using the specified transforms.
        __getitem__(idx): Returns the transformed point cloud and its label at the given index.
        get_classes(): Returns the class index mapping.
    """
    def __init__(self, root_dir, valid=False, folder="train", transform=default_transforms()):
        """
        Initializes the dataset with a directory path, and optionally applies transformations.
        Parameters:
            root_dir (str or Path): The directory with all the point cloud files.
            valid (bool): If True, will use a separate set of transforms for validation data.
            folder (str): The subdirectory within root_dir that contains the point cloud files.
            transform (callable): The transformations to apply to each point cloud.
        """
        self.root_dir = root_dir
        # List all the directories and create a mapping from folder names to class indices
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        # Use the provided transform or default to the standard transforms
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
        # Populate the list of files in the dataset
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    self.files.append({'pcd_path': new_dir/file, 'category': category})

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.files)

    def __preproc__(self, file):
        """
        Reads and processes a file using the given transformations.
        Parameters:
            file: The file to process.
        Returns:
            The transformed point cloud data.
        """
        verts, faces = read_file(file)
        pointcloud = self.transforms((verts, faces)) if self.transforms else (verts, faces)
        return pointcloud

    def __getitem__(self, idx):
        """
        Retrieves a sample and its label based on the index.
        Parameters:
            idx (int): The index of the sample to retrieve.
        Returns:
            dict: A dictionary containing the transformed point cloud and its associated class label.
        """
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return {'pointcloud': pointcloud, 'category': self.classes[category]}

    def get_classes(self):
        """
        Returns the dictionary mapping classes to indices.
        """
        return self.classes

class RandRotation_z(object):
    """
    Applies a random rotation around the Z-axis to the point cloud.
    """
    def __call__(self, pointcloud):
        """
        Applies the transformation to the given point cloud.
        Parameters:
            pointcloud (numpy.ndarray): A 2D array representing the point cloud to be rotated.
        Returns:
            numpy.ndarray: The rotated point cloud.
        """
        # Verify pointcloud is a 2D array
        assert len(pointcloud.shape) == 2, "Point cloud must be a 2D array."

        # Generate a random rotation angle
        theta = random.random() * 2. * math.pi
        # Create a rotation matrix for the Z-axis
        rot_matrix = np.array([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1]
        ])
        # Apply the rotation to the point cloud
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud

class RandomNoise(object):
    """
    Adds random Gaussian noise to each point in the point cloud.
    """
    def __call__(self, pointcloud):
        """
        Applies Gaussian noise to the given point cloud.
        Parameters:
            pointcloud (numpy.ndarray): A 2D array representing the point cloud to be noised.
        Returns:
            numpy.ndarray: The point cloud with added Gaussian noise.
        """
        # Verify pointcloud is a 2D array
        assert len(pointcloud.shape) == 2, "Point cloud must be a 2D array."

        # Generate Gaussian noise
        noise = np.random.normal(0, 0.02, pointcloud.shape)
        # Add the noise to the point cloud
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud

def get_transforms():
    """
    Creates a composite transformation for point cloud data preprocessing,
    including sampling, normalization, random rotation, noise addition, and tensor conversion.
    Returns:
        torchvision.transforms.Compose: A composition of point cloud data transformations.
    """
    return transforms.Compose([
        PointSampler(1024),   # Sample 1024 points from the point cloud
        Normalize(),          # Normalize the point cloud
        RandRotation_z(),     # Apply a random rotation around the Z-axis
        RandomNoise(),        # Add random Gaussian noise to the point cloud
        ToTensor()            # Convert the point cloud to a PyTorch tensor
    ])

