# PointNet

This is a PyTorch implementation of PointNet as described in the paper [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593).

_____
### Semester project for ACIT4030

----
### Requirements
- Python 3.11
- PyTorch 2.0.1
- Poetry
```bash 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Then run
```bash
poetry install
```

### Dataset
- ModelNet10 dataset

```bash
wget http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
unzip -q ModelNet10.zip
```

## Done
Pointnet should work