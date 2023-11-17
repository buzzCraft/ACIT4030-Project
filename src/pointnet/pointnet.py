import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Tnet(nn.Module):
    """
    The T-net is a mini PointNet that outputs a transformation matrix
    mini neural network architecture from the PointNet family, which
    is specifically designed to compute a transformation matrix for spatial
    data points. This transformation matrix is used to align the input point cloud
    to a canonical space, improving the subsequent processing steps.

    The network uses 1D convolutional layers followed by fully connected layers
    to regress to the elements of the transformation matrix.

    Attributes:
        k (int): The size of the input point set (i.e., number of dimensions).
                 Typically, k=3 for 3D point cloud data.
        conv1 (nn.Conv1d): First convolutional layer that expands the feature
                           dimension from k to 64.
        conv2 (nn.Conv1d): Second convolutional layer that expands the feature
                           dimension from 64 to 128.
        conv3 (nn.Conv1d): Third convolutional layer that expands the feature
                           dimension to 1024.
        fc1 (nn.Linear): First fully connected layer that reduces the feature
                         dimension from 1024 to 512.
        fc2 (nn.Linear): Second fully connected layer that reduces the feature
                         dimension from 512 to 256.
        fc3 (nn.Linear): Third fully connected layer that outputs k*k elements,
                         which are reshaped to form the k-by-k transformation matrix.
        bn1, bn2, bn3, bn4, bn5 (nn.BatchNorm1d): Batch normalization layers
                                                   corresponding to each
                                                   convolutional and fully
                                                   connected layer.

    Methods:
        forward(input): Defines the computation performed at every call.
                        Takes an input tensor representing a batch of point sets
                        and returns the transformation matrix for each set.
    """

    def __init__(self, k=3):
        """
        Initializes the T-net module.

        Parameters:
            k (int): The size of each input point set, default is 3 (for 3D points).
        """
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input):
        """
        Defines the forward pass of the T-net module.

        Parameters:
            input (torch.Tensor): The input tensor containing a batch of point
                                  sets with shape (batch_size, n, k), where n is
                                  the number of points in each set, and k is the
                                  number of dimensions for each point.

        Returns:
            torch.Tensor: A batch of transformation matrices with shape
                          (batch_size, k, k).
        """
        # input.shape == (batch_size, n, k)
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))

        # initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1)
        if xb.is_cuda:
            init = init.cuda()
        # The output matrix is the sum of the identity matrix and the learned matrix
        matrix = self.fc3(xb).view(-1, self.k, self.k) + init
        return matrix


class Transform(nn.Module):
    """
    Transform is a network module that applies a series of transformations
    to the input point cloud data for feature learning. It consists of an input
    transformation, a feature transformation, and several convolutional layers for
    feature extraction.

    Attributes:
        input_transform (Tnet): A T-net module that outputs a 3x3 transformation
                                matrix used to align the input point cloud in a
                                canonical space.
        feature_transform (Tnet): A T-net module that outputs a 64x64 transformation
                                  matrix to further align features after the first
                                  convolutional layer.
        conv1 (nn.Conv1d): The first convolutional layer with a kernel size of 1
                           that expands the feature dimension from 3 to 64.
        conv2 (nn.Conv1d): The second convolutional layer with a kernel size of 1
                           that expands the feature dimension from 64 to 128.
        conv3 (nn.Conv1d): The third convolutional layer with a kernel size of 1
                           that expands the feature dimension to 1024.
        bn1 (nn.BatchNorm1d): Batch normalization layer for the output of conv1.
        bn2 (nn.BatchNorm1d): Batch normalization layer for the output of conv2.
        bn3 (nn.BatchNorm1d): Batch normalization layer for the output of conv3.

    Methods:
        forward(input): Defines the computation performed at every call.
                        Applies the input and feature transformations, and
                        extracts features using convolutional layers.
    """

    def __init__(self):
        super().__init__()
        # Define the input transformation (3x3 matrix) module
        self.input_transform = Tnet(k=3)

        # Define the feature transformation (64x64 matrix) module
        self.feature_transform = Tnet(k=64)

        # Define convolutional layers and batch normalization layers
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, input):
        """
        Forward pass for the Transform module.

        Parameters:
            input (torch.Tensor): The input tensor containing a batch of point
                                  sets with shape (batch_size, n, 3), where n is
                                  the number of points.

        Returns:
            tuple: A tuple containing the output feature tensor, the input
                   transformation matrix, and the feature transformation matrix.
        """
        # Apply the input transformation and multiply it with the input batch
        matrix3x3 = self.input_transform(input)
        xb = torch.bmm(torch.transpose(input, 1, 2), matrix3x3).transpose(1, 2)

        # Apply the first convolutional layer
        xb = F.relu(self.bn1(self.conv1(xb)))

        # Apply the feature transformation and multiply it with the feature batch
        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb, 1, 2), matrix64x64).transpose(1, 2)

        # Apply the subsequent convolutional layers
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))

        # Apply max pooling to extract the most significant features
        xb = nn.MaxPool1d(xb.size(-1))(xb)

        # Flatten the features to vectorize them for further processing
        output = nn.Flatten(1)(xb)

        # Return the output feature tensor and the transformation matrices
        return output, matrix3x3, matrix64x64


class PointNet(nn.Module):
    """
    PointNet is a deep neural network designed to perform classification
    (and segmentation) tasks on point cloud data.

    Attributes:
        transform (Transform): An instance of the Transform class which applies
                               transformations to the input data and extracts features.
        fc1 (nn.Linear): The first fully connected layer that reduces the feature
                         dimension from 1024 to 512.
        fc2 (nn.Linear): The second fully connected layer that reduces the feature
                         dimension from 512 to 256.
        fc3 (nn.Linear): The third fully connected layer that outputs the number
                         of classes for classification.
        bn1 (nn.BatchNorm1d): Batch normalization layer for the output of fc1.
        bn2 (nn.BatchNorm1d): Batch normalization layer for the output of fc2.
        dropout (nn.Dropout): Dropout layer for regularization, with a dropout
                              probability of 0.3.
        logsoftmax (nn.LogSoftmax): LogSoftmax layer to apply to the final output,
                                    providing log-probabilities for classification.

    Methods:
        forward(input): Defines the forward pass of the PointNet module.
    """

    def __init__(self, classes=10):
        """
        Initializes the PointNet module.

        Parameters:
            classes (int): The number of classes for classification. The default is 10.
        """
        super().__init__()
        # The Transform module to extract features from raw point cloud data
        self.transform = Transform()

        # Fully connected layers to perform classification
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, classes)

        # LogSoftmax for converting outputs to log-probabilities
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        """
        Forward pass for the PointNet module.

        Parameters:
            input (torch.Tensor): The input tensor containing a batch of point sets.

        Returns:
            tuple: A tuple containing the log-probabilities of class predictions,
                   the input transformation matrix, and the feature transformation matrix.
        """
        # Extract features from the input data
        xb, matrix3x3, matrix64x64 = self.transform(input)

        # Apply the fully connected layers with batch normalization, dropout, and ReLU activation
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))

        # Obtain the class scores before softmax
        output = self.fc3(xb)

        # Return the log-probabilities along with the transformation matrices
        return self.logsoftmax(output), matrix3x3, matrix64x64


def pointnetloss(outputs, labels, m3x3, m64x64, alpha=0.0001):
    """
    Calculate the PointNet loss function which combines a negative log likelihood loss
    for classification with a regularization term that encourages the transformation
    matrices to be orthogonal. This regularization helps the network to preserve the
    geometric structure of the input data.

    Parameters:
        outputs (torch.Tensor): The log-probabilities of the class predictions from the
                                PointNet model, with shape (batch_size, num_classes).
        labels (torch.Tensor): The ground-truth labels for the input data, with shape
                               (batch_size,).
        m3x3 (torch.Tensor): The transformation matrix for input data, with shape
                             (batch_size, 3, 3).
        m64x64 (torch.Tensor): The transformation matrix for features, with shape
                               (batch_size, 64, 64).
        alpha (float): The weight given to the regularization term in the loss function.
                       Default value is 0.0001.

    Returns:
        torch.Tensor: The total loss for the batch, which is the sum of the negative
                      log likelihood loss and the regularization term.
    """
    # Define the classification loss function
    criterion = torch.nn.NLLLoss()

    labels = labels.long()

    # Get the batch size from the outputs tensor
    bs = outputs.size(0)

    # Create identity matrices with gradients enabled, repeated for the entire batch
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)

    # If the outputs are on GPU, ensure the identity matrices are also on GPU
    if outputs.is_cuda:
        id3x3 = id3x3.cuda()
        id64x64 = id64x64.cuda()

    # Calculate the difference between the identity matrix and the product of
    # the transformation matrix and its transpose, for both 3x3 and 64x64 matrices
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))

    # Calculate the Frobenius norm (torch.norm) of the difference tensors, which
    # serves as the regularization term encouraging orthogonality of the transformation
    # matrices. The norms are then averaged over the batch size and weighted by alpha.
    reg_loss = alpha * (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(bs)

    # The total loss is the sum of the classification loss and the regularization term
    return criterion(outputs, labels) + reg_loss
