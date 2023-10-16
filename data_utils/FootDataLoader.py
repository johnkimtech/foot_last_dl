import os
import warnings
import pandas as pd
from torch.utils.data import Dataset
from data_utils.helpers import (
    read_point_cloud_text,
    random_point_dropout,
    shift_point_cloud,
    pc_normalize,
    stl_to_xyz_with_normals_vectorized,
)

warnings.filterwarnings("ignore")
foot_labels_cols = ["발 길이 ", "발볼 둘레 ", "발등 둘레", "발 뒤꿈치 둘레", "발가락 둘레"]


class FootDataLoader(Dataset):
    def __init__(
        self,
        root: str = None,
        num_points: int = 6000,
        use_normals: bool = False,
        split="train",
        infer_data_csv: str = None,  # CSV files containing
    ):
        self.npoints = num_points
        self.use_normals = use_normals

        assert split in ["train", "test", "infer"], f"Unsupported split: {split}"

        self.split = split

        if split != "infer":
            assert (
                root is not None
            ), "Root data folder must be provided in training or testing mode"
            data_path = os.path.join(root, f"{split}.csv")
            self.df = pd.read_csv(data_path)
        else:
            self.df = pd.read_csv(infer_data_csv)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        foot = self.df.iloc[index]

        pc_path = foot["3D"]

        _, pc_ext = os.path.splitext(pc_path)

        if pc_ext.lower() == ".txt":
            points = read_point_cloud_text(
                pc_path, flip_axis=1 if foot["Foot"] == "R" else -1
            )
        elif pc_ext.lower() == ".stl":
            # Only use STL in inference / testing, not in TRAINING
            points = stl_to_xyz_with_normals_vectorized(
                pc_path,
                stride=10,
                flip_axis=1 if foot["Foot"] == "R" else -1,
                with_normals=self.use_normals,
                permutate=True,
            )
            # print('Converted', points.shape)
        else:
            raise (f"Unsupported data file extension: {pc_ext}")

        dims = 6 if self.use_normals else 3
        points = points[: self.npoints, :dims].astype(float)

        points[:, 0:3], scale = pc_normalize(points[:, 0:3], return_scale=True)

        if self.split == "train":
            # data augmentation
            points = random_point_dropout(points)
            points = shift_point_cloud(points)

        if self.split == "infer":
            return points, scale, foot["No."]

        labels = foot[foot_labels_cols].to_numpy().astype(float)
        labels_normalized = labels / scale

        return points, labels_normalized, scale
