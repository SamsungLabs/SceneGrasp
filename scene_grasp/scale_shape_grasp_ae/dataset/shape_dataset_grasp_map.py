"""
This file loads shape dataset from the nocs-grasp-dataset generated using our own
internal grasp generation tool.

# Ok. Here is the action plan:
# - The class will take in nocs root and nocs-grasp data root
# - For every pickle file, we will create a datapoint.
# - when the datapoint is returned: we will basically load the point-cloud in the
#   unit-canonical space. However, the 
"""

from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
from common.config.config_dataset_details import get_nocs_information_dicts


class ShapeDatasetGraspMap(data.Dataset):
    def __init__(
        self,
        nocs_grasp_root,
        mode,
        n_points=2048,
        augment=False,
        category_id: str = None,
        return_cat_id: bool = False,  # return cat id in target
        return_datafile_ind: bool = False,  # return index in the data-file-path array
    ):
        assert mode == "train" or mode == "val", 'Mode must be "train" or "val".'
        assert not augment, "augmentation is not implemented yet"

        self.mode = mode
        self.dataset_root = Path(nocs_grasp_root) / "obj_models" / self.mode
        print("Dataset root: ", self.dataset_root)
        self.n_points = n_points
        self.augment = augment
        self.category_id = category_id
        self.return_cat_id = return_cat_id
        _, shapenet_id_to_cat_id, _ = get_nocs_information_dicts()
        self.data_files = []
        self.cat_ids = []
        self.return_datafile_ind = return_datafile_ind
        for data_file_path in self.dataset_root.rglob(
            "network_train_grasp_parameter_data_without_fps_at_scale_*.pkl"
        ):
            self.data_files.append(data_file_path)
            if self.return_cat_id:
                cat_id = shapenet_id_to_cat_id[data_file_path.parent.parent.name]
                self.cat_ids.append(cat_id)
        self.length = len(self.data_files)

    def __len__(self):
        return self.length

    @staticmethod
    def read_data_file(
        data_file_path,
        n_points,
    ):
        with open(data_file_path, "rb") as fp:
            data = pickle.load(fp)

        # randomly downsample
        xyz = data["xyz"].astype(np.float32)
        current_num_points = len(xyz)
        assert current_num_points >= n_points, "Not enough points in shape."
        if not current_num_points > n_points:
            downsample_indices = np.random.choice(current_num_points, n_points)
            xyz = xyz[downsample_indices, :]
        else:
            downsample_indices = np.arange(current_num_points)

        scale = data["scale"].astype(np.float32)
        target = {
            "success": data["success"][downsample_indices].astype(np.float32),
            "grasp_width": data["grasp_width"][downsample_indices].astype(np.float32),
            "grasp_width_one_hot": data["grasp_width_one_hot"][
                downsample_indices
            ].astype(np.float32),
            "approach_dir": data["approach_dirs"][downsample_indices].astype(
                np.float32
            ),
            "baseline_dir": data["baseline_dirs"][downsample_indices].astype(
                np.float32
            ),
        }

        return {"xyz": xyz, "scale": np.array([scale])}, target

    def __getitem__(self, index):
        data_file_path = self.data_files[index]
        inp, target = self.read_data_file(
            data_file_path, self.n_points
        )
        if self.return_cat_id:
            target["cat_id"] = self.cat_ids[index]
        if self.return_datafile_ind:
            target["datafile_ind"] = index

        
        return inp, target
