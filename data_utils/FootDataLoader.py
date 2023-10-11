import os
import warnings
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_utils.helpers import read_point_cloud_text, random_point_dropout,shift_point_cloud,pc_normalize

warnings.filterwarnings('ignore')
foot_labels_cols = ['발 길이 ', '발볼 둘레 ', '발등 둘레', '발 뒤꿈치 둘레', '발가락 둘레']

class FootDataLoader(Dataset):
    def __init__(self, root, num_points, use_normals, split='train'):
        self.npoints = num_points
        self.use_normals = use_normals

        assert (split == 'train' or split == 'test')
        self.split = split
        self.df = pd.read_csv(f"{root}/{split}.csv")
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        foot = self.df.iloc[index]
        labels = foot[foot_labels_cols].to_numpy().astype(float)

        pc_path = foot["3D"]
        points = read_point_cloud_text(
            pc_path, 
            flip_axis=1 if foot["Foot"] == "R" else -1
        )
        
        dims = 6 if self.use_normals else 3
        points = points[:self.npoints,:dims].astype(float)

        points[:, 0:3], scale = pc_normalize(points[:, 0:3], return_scale=True)

        if self.split == "train":
            # data augmentation
            points = random_point_dropout(points)
            points = shift_point_cloud(points)

        labels_normalized = labels / scale

        return points, labels_normalized, scale