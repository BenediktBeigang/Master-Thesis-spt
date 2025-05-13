# from src.models.semantic import SemanticSegmentationModule
# from src.datamodules.kitti360 import KITTI360DataModule
# from pytorch_lightning import Trainer

# # Predict behavior for semantic segmentation from a torch DataLoader
# dataloader = KITTI360DataModule(...)
# model = SemanticSegmentationModule(...)
# trainer = Trainer(...)
# batch, output = trainer.predict(model=model, dataloaders=dataloader)

import os
import sys

# Add the project's files to the python path
# file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # for .py script
file_path = os.path.dirname(os.path.abspath(''))  # for .ipynb notebook
sys.path.append(file_path)

import hydra
from src.utils import init_config
import torch
from src.transforms import *
from src.utils.widgets import *
from src.data import *

device_widget = make_device_widget()
task_widget, expe_widget = make_experiment_widgets()
split_widget = make_split_widget()
ckpt_widget = make_checkpoint_file_search_widget()

##############################################################################
######### Select your device, experiment, split, and pretrained model ########
##############################################################################

# Summarizing selected task, experiment, split, and checkpoint
print(f"You chose:")
print(f"  - device={device_widget.value}")
print(f"  - task={task_widget.value}")
print(f"  - split={split_widget.value}")
print(f"  - experiment={expe_widget.value}")
print(f"  - ckpt={ckpt_widget.value}")

##############################################################################
######################### Parsing the config files ###########################
##############################################################################

# Parse the configs using hydra
cfg = init_config(overrides=[
    f"experiment={task_widget.value}/{expe_widget.value}",
    f"ckpt_path={ckpt_widget.value}",
    f"datamodule.load_full_res_idx={True}"  # only when you need full-resolution predictions 
])

##############################################################################
##################### Datamodule and model instantiation #####################
##############################################################################

# Instantiate the datamodule
datamodule = hydra.utils.instantiate(cfg.datamodule)
datamodule.prepare_data()
datamodule.setup()

# Pick among train, val, and test datasets. It is important to note that
# the train dataset produces augmented spherical samples of large 
# scenes, while the val and test dataset load entire tiles at once
if split_widget.value == 'train':
    dataset = datamodule.train_dataset
elif split_widget.value == 'val':
    dataset = datamodule.val_dataset
elif split_widget.value == 'test':
    dataset = datamodule.test_dataset
else:
    raise ValueError(f"Unknown split '{split_widget.value}'")

# Print a summary of the datasets' classes
dataset.print_classes()

# Instantiate the model
model = hydra.utils.instantiate(cfg.model)

# Load pretrained weights from a checkpoint file
if ckpt_widget.value is not None:
    model = model._load_from_checkpoint(cfg.ckpt_path)

# Move model to selected device
model = model.eval().to(device_widget.value)

##############################################################################
################ Hierarchical partition loading and inference ################
##############################################################################

# For the sake of visualization, we require that NAGAddKeysTo does not 
# remove input Data attributes after moving them to Data.x, so we may 
# visualize them
for t in dataset.on_device_transform.transforms:
    if isinstance(t, NAGAddKeysTo):
        t.delete_after = False

# Load the first dataset item. This will return the hierarchical 
# partition of an entire tile, as a NAG object 
nag = dataset[0]

# Apply on-device transforms on the NAG object. For the train dataset, 
# this will select a spherical sample of the larger tile and apply some
# data augmentations. For the validation and test datasets, this will
# prepare an entire tile for inference
nag = dataset.on_device_transform(nag.to(device_widget.value))

# Inference, returns a task-specific ouput object carrying predictions
with torch.no_grad():
    output = model(nag)

# Compute the level-0 (voxel-wise) semantic segmentation predictions 
# based on the predictions on level-1 superpoints and save those for 
# visualization in the level-0 Data under the 'semantic_pred' attribute
nag[0].semantic_pred = output.voxel_semantic_pred(super_index=nag[0].super_index)

# Similarly, compute the level-0 panoptic segmentation predictions, if 
# relevant
if task_widget.value == 'panoptic':
    vox_y, vox_index, vox_obj_pred = output.voxel_panoptic_pred(super_index=nag[0].super_index)
    nag[0].obj_pred = vox_obj_pred
    



# Compute the full-resolution semantic prediction. These labels are ordered 
# with respect to the full-resolution data points in the corresponding raw 
# input file. Note that we do not provide the pipeline for recovering the 
# corresponding full-resolution positions, colors, etc. 
raw_semseg_y = output.full_res_semantic_pred(
    super_index_level0_to_level1=nag[0].super_index,
    sub_level0_to_raw=nag[0].sub)

# Similarly, we can compute the full-resolution panoptic prediction. 
# The returned outputs are (in order) the predicted semantic prediction, the
# predicted instance index, and the InstancData object holding this information 
# under another format
if task_widget.value == 'panoptic':
    raw_pano_y, raw_index, raw_obj_pred = output.full_res_panoptic_pred(
        super_index_level0_to_level1=nag[0].super_index, 
        sub_level0_to_raw=nag[0].sub)


##############################################################################
######################### Visualizing an entire tile #########################
##############################################################################

# Visualize the hierarchical partition
nag.show( 
    class_names=dataset.class_names,
    class_colors=dataset.class_colors,
    stuff_classes=dataset.stuff_classes,
    num_classes=dataset.num_classes,
    max_points=100000
)
