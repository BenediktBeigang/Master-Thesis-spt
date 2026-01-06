import os
import sys
import glob
import torch
import shutil
import logging
from zipfile import ZipFile
from plyfile import PlyData
import laspy
from typing import List
from torch_geometric.data.extract import extract_zip
from torch_geometric.nn.pool.consecutive import consecutive_cluster

from src.datasets import BaseDataset
from src.data import Data, InstanceData
from src.datasets.synthetic_config import *
from src.utils.neighbors import knn_2
from src.utils.color import to_float_rgb


DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)


# Occasional Dataloader issues with KITTI360 on some machines. Hack to
# solve this:
# https://stackoverflow.com/questions/73125231/pytorch-dataloaders-bad-file-descriptor-and-eof-for-workers0
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


__all__ = ['Synthetic']


########################################################################
#                                 Utils                                #
########################################################################

def read_synthetic(
        filepath: str,
        xyz: bool = True,
        rgb: bool = True,
        semantic: bool = True,
        remap: bool = False
) -> Data:
    """Read a synthetic pointcloud saved as LAS.

    :param filepath: str
        Absolute path to the LAS file
    :param xyz: bool
        Whether XYZ coordinates should be saved in the output Data.pos
    :param rgb: bool
        Whether RGB colors should be saved in the output Data.rgb
    :param semantic: bool
        Whether semantic labels should be saved in the output Data.y
    :param remap: bool
        Whether semantic labels should be mapped from their KITTI-360 ID
        to their train ID. For more details, see:
        https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/evalPointLevelSemanticLabeling.py
    """
    data = Data()
    print("Reading synthetic tile from", filepath)
    
    las = laspy.read(filepath)
    
    if xyz:
        pos = torch.stack([
            torch.tensor(las[axis].copy(), dtype=torch.float32)
            for axis in ["X", "Y", "Z"]
        ], dim=-1)
        pos *= las.header.scale  # LAS scale anwenden
        pos_offset = pos[0]
        data.pos = (pos - pos_offset).float()
        data.pos_offset = pos_offset

    if rgb:
        # RGB in LAS ist typischerweise uint16 in [0, 65535]
        data.rgb = to_float_rgb(torch.stack([
            torch.FloatTensor(las[axis].astype('float32') / 65535)
            for axis in ["red", "green", "blue"]], dim=-1))

    if semantic:
        # Anpassen je nach deinem LAS-Attribut für Labels
        y = torch.from_numpy(las['point_source_id'].astype('int64'))
        data.y = torch.from_numpy(ID2TRAINID)[y] if remap else y

    return data


########################################################################
#                               KITTI360                               #
########################################################################

class Synthetic(BaseDataset):
    """Synthetic dataset.

    Dataset website: http://www.cvlibs.net/datasets/kitti-360/

    Parameters
    ----------
    root : `str`
        Root directory where the dataset should be saved.
    stage : {'train', 'val', 'test', 'trainval'}
    transform : `callable`
        transform function operating on data.
    pre_transform : `callable`
        pre_transform function operating on data.
    pre_filter : `callable`
        pre_filter function operating on data.
    on_device_transform: `callable`
        on_device_transform function operating on data, in the
        'on_after_batch_transfer' hook. This is where GPU-based
        augmentations should be, as well as any Transform you do not
        want to run in CPU-based DataLoaders
    """

    @property
    def class_names(self) -> List[str]:
        """List of string names for dataset classes. This list must be
        one-item larger than `self.num_classes`, with the last label
        corresponding to 'void', 'unlabelled', 'ignored' classes,
        indicated as `y=self.num_classes` in the dataset labels.
        """
        return CLASS_NAMES

    @property
    def num_classes(self) -> int:
        """Number of classes in the dataset. Must be one-item smaller
        than `self.class_names`, to account for the last class name
        being used for 'void', 'unlabelled', 'ignored' classes,
        indicated as `y=self.num_classes` in the dataset labels.
        """
        return SYNTHETIC_NUM_CLASSES

    @property
    def stuff_classes(self) -> List[int]:
        """List of 'stuff' labels for INSTANCE and PANOPTIC
        SEGMENTATION (setting this is NOT REQUIRED FOR SEMANTIC
        SEGMENTATION alone). By definition, 'stuff' labels are labels in
        `[0, self.num_classes-1]` which are not 'thing' labels.

        In instance segmentation, 'stuff' classes are not taken into
        account in performance metrics computation.

        In panoptic segmentation, 'stuff' classes are taken into account
        in performance metrics computation. Besides, each cloud/scene
        can only have at most one instance of each 'stuff' class.

        IMPORTANT:
        By convention, we assume `y ∈ [0, self.num_classes-1]` ARE ALL
        VALID LABELS (i.e. not 'ignored', 'void', 'unknown', etc), while
        `y < 0` AND `y >= self.num_classes` ARE VOID LABELS.
        """
        return STUFF_CLASSES

    @property
    def class_colors(self) -> List[List[int]]:
        """Colors for visualization, if not None, must have the same
        length as `self.num_classes`. If None, the visualizer will use
        the label values in the data to generate random colors.
        """
        return CLASS_COLORS

    @property
    def all_base_cloud_ids(self) -> List[str]:
        """Dictionary holding lists of paths to the clouds, for each
        stage.

        The following structure is expected:
            `{'train': [...], 'val': [...], 'test': [...]}`
        """
        return TILES

    def download_dataset(self) -> None:
        """Download the Synthetic dataset.
        """
        log.info("The synthetic dataset does not support automatic download. You have to place the dataset by hand.\n")
        

    def read_single_raw_cloud(self, raw_cloud_path: str) -> 'Data':
        """Read a single raw cloud and return a `Data` object, ready to
        be passed to `self.pre_transform`.

        This `Data` object should contain the following attributes:
          - `pos`: point coordinates
          - `y`: OPTIONAL point semantic label
          - `obj`: OPTIONAL `InstanceData` object with instance labels
          - `rgb`: OPTIONAL point color
          - `intensity`: OPTIONAL point LiDAR intensity

        IMPORTANT:
        By convention, we assume `y ∈ [0, self.num_classes-1]` ARE ALL
        VALID LABELS (i.e. not 'ignored', 'void', 'unknown', etc),
        while `y < 0` AND `y >= self.num_classes` ARE VOID LABELS.
        This applies to both `Data.y` and `Data.obj.y`.
        """
        return read_synthetic(raw_cloud_path, xyz=True, rgb=True, semantic=True, remap=True)

    @property
    def raw_file_structure(self) -> str:
        return f"""
    {self.root}/
        └── raw/
            └── {{seq:0>4}}_{{seq:0>12}}_frames_1_to_{{seq:0>4}}_noise_parts_processed.las
        """

    def id_to_relative_raw_path(self, id: str) -> str:
        """All files are directly in the raw/ folder"""
        return self.id_to_base_id(id) + '.las'

    
    def processed_to_raw_path(self, processed_path: str) -> str:
        """Return the raw cloud path corresponding to the input processed path."""
        # Extract cloud_id from processed path
        stage, hash_dir, cloud_id = \
            osp.splitext(processed_path)[0].split(os.sep)[-3:]
        
        # Remove tiling from cloud_id if any
        base_cloud_id = self.id_to_base_id(cloud_id)
        
        # Alle raw files sind direkt im raw_dir
        raw_path = osp.join(self.raw_dir, base_cloud_id + '.las')
        return raw_path

