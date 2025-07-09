from __future__ import annotations
import numpy as np
import os.path as osp

USE_KITTI_TRAIN_IDS: bool = False
ONLY_PIPES: bool = True


########################################################################
#                              Data splits                             #
########################################################################

TILES: dict[str, list[str]] = {
    'train': [
        '0010_202506081336_frames_1_to_1056_noise_parts',
        '0011_202506081407_frames_1_to_1056_noise_parts',
        '0013_202506081442_frames_1_to_1056_noise_parts',
        '0014_202506130743_frames_1_to_1056_noise_parts',
        '0015_202506130819_frames_1_to_1056_noise_parts',
        '0016_202506130842_frames_1_to_1056_noise_parts',
        '0017_202506130906_frames_1_to_1056_noise_parts',
        '0018_202506130931_frames_1_to_1056_noise_parts',
        '0019_202506131118_frames_1_to_1056_noise_parts',
        '0020_202506131140_frames_1_to_1056_noise_parts',
        '0021_202506131216_frames_1_to_1056_noise_parts',
        '0022_202506131253_frames_1_to_1056_noise_parts',
        '0023_202506230623_frames_1_to_1056_noise_parts',
        '0024_202506230655_frames_1_to_1056_noise_parts',
        '0025_202506230727_frames_1_to_1056_noise_parts',
        '0026_202506230751_frames_1_to_1056_noise_parts',
        '0027_202506230820_frames_1_to_1056_noise_parts',
        '0028_202506230935_frames_1_to_1056_noise_parts',
        '0029_202506231006_frames_1_to_1056_noise_parts',
        '0030_202506210952_frames_1_to_1056_noise_parts',
        '0031_202506240609_frames_1_to_1056_noise_parts',
        '0032_202506240636_frames_1_to_1056_noise_parts',
        '0033_202506240705_frames_1_to_1056_noise_parts',
        '0034_202506240733_frames_1_to_1056_noise_parts',
        '0035_202506240804_frames_1_to_1056_noise_parts',
        '0036_202506240835_frames_1_to_1056_noise_parts',
        '0037_202506240910_frames_1_to_1056_noise_parts',
        '0038_202506240940_frames_1_to_1056_noise_parts',
        '0039_202506241010_frames_1_to_1056_noise_parts',
        '0040_202506241038_frames_1_to_1056_noise_parts',
        '0041_202506241135_frames_1_to_1056_noise_parts',
        '0042_202506241210_frames_1_to_1056_noise_parts',
        '0043_202506021352_frames_1_to_1049_noise_parts',
        '0044_202506241317_frames_1_to_1056_noise_parts',
        '0045_202506021650_frames_1_to_1049_noise_parts',
        '0046_202507030715_frames_1_to_1056_noise_parts',
        '0047_202507030745_frames_1_to_1056_noise_parts',
        '0048_202507030823_frames_1_to_1056_noise_parts',
        '0049_202507030900_frames_1_to_1056_noise_parts',
        '0050_202507031322_frames_1_to_1056_noise_parts',
        '0051_202507031356_frames_1_to_1056_noise_parts',
        '0052_202507031427_frames_1_to_1056_noise_parts',
        '0053_202507040819_frames_1_to_1056_noise_parts',
        '0054_202507040855_frames_1_to_1056_noise_parts',
        '0055_202507040923_frames_1_to_1056_noise_parts',
        '0056_202507040952_frames_1_to_1056_noise_parts',
        '0057_202507041023_frames_1_to_1056_noise_parts',
        '0058_202507041057_frames_1_to_1056_noise_parts',
        '0059_202507041124_frames_1_to_1056_noise_parts',
        '0060_202507041157_frames_1_to_1056_noise_parts',
        '0061_202507041225_frames_1_to_1056_noise_parts',
        '0062_202507041245_frames_1_to_1056_noise_parts',
    ],
    
    'val': [
        '0063_202507041310_frames_1_to_1056_noise_parts',
        '0064_202507041340_frames_1_to_1056_noise_parts',
        '0100_202506250855_frames_1_to_1056_noise_parts',
        '0101_202506250908_frames_1_to_1056_noise_parts',
        '0102_202506250921_frames_1_to_1056_noise_parts',
        '0103_202506250937_frames_1_to_1056_noise_parts',
        '0104_202506250950_frames_1_to_1056_noise_parts',
        '0105_202506251004_frames_1_to_1056_noise_parts',
        '0106_202506251017_frames_1_to_1056_noise_parts',
        '0107_202506251031_frames_1_to_1056_noise_parts',
        '0108_202506251046_frames_1_to_1056_noise_parts',
        '0109_202506251059_frames_1_to_1056_noise_parts',
        '0110_202507031442_frames_1_to_1056_noise_parts',
    ],
    
    'test': [],
    
    # 'test': [
    #     'ontras_0',
    #     'ontras_1',
    #     'ontras_2',
    #     'ontras_3',
    #     'ontras_4',
    #     'ontras_5',
    # ]
}


########################################################################
#                                Labels                                #
########################################################################

SYNTHETIC_NUM_CLASSES: int = 15 if USE_KITTI_TRAIN_IDS else (2 if ONLY_PIPES else 5)

def remapArrays() -> np.ndarray | None:
    if not USE_KITTI_TRAIN_IDS and not ONLY_PIPES:
        print("When USE_KITTI_TRAIN_IDS and ONLY_PIPES are both False this mapping should not be called!")
        return None
    if not USE_KITTI_TRAIN_IDS and ONLY_PIPES:
        return np.asarray([0, 1, 1, 1, 1, 0])
    if USE_KITTI_TRAIN_IDS and not ONLY_PIPES:
        # targeting kitti kitti-trainIDs [terrain/9, fence/4, pole/5, car/11, motorcycle/13], is only used if USE_KITTI_TRAIN_IDS is True
        return np.asarray([9, 4, 5, 11, 13, 8])
    return np.asarray([9, 4, 4, 4, 4, 8])

ID2TRAINID: np.ndarray | None =  remapArrays()

############################
####### CLASS NAMES ########
############################

CLASS_NAMES_SYNTHETIC: list[str] = [
    'ground',    # 0
    'pipe',      # 1
    'armature',  # 2
    'corner',    # 3
    'adapter',   # 4
    'other']     # 5

# Erweitere CLASS_NAMES für alle 15 KITTI-360 Klassen
CLASS_NAMES_KITTI: list[str] = [
    'road',           # 0
    'sidewalk',       # 1  
    'building',       # 2
    'wall',           # 3
    'fence',          # 4 -> deine 'pipe' Klasse
    'pole',           # 5 -> deine 'armature' Klasse
    'traffic light',  # 6
    'traffic sign',   # 7
    'vegetation',     # 8
    'terrain',        # 9 -> deine 'ground' Klasse
    'sky',            # 10
    'car',            # 11 -> deine 'corner' Klasse
    'truck',          # 12
    'motorcycle',     # 13 -> deine 'adapter' Klasse
    'other',          # 14
    'ignored']        # 15 -> void/ignored Klasse

CLASS_NAMES: list[str] = CLASS_NAMES_KITTI if USE_KITTI_TRAIN_IDS else CLASS_NAMES_SYNTHETIC

#############################
####### CLASS COLORS ########
#############################

CLASS_COLORS_KITTI: np.ndarray = np.asarray([
    (  0,  0,  0),
    (  0,  0,  0),
    (  0,  0,  0),
    (  0,  0,  0),
    (  0,  0,  0),
    (111, 74,  0),
    ( 81,  0, 81),
    (128, 64,128),
    (244, 35,232),
    (250,170,160),
    (230,150,140),
    ( 70, 70, 70),
    (102,102,156),
    (190,153,153),
    (180,165,180),
    (150,100,100),
    (150,120, 90),
    (153,153,153),
    (153,153,153),
    (250,170, 30),
    (220,220,  0),
    (107,142, 35),
    (152,251,152),
    ( 70,130,180),
    (220, 20, 60),
    (255,  0,  0),
    (  0,  0,142),
    (  0,  0, 70),
    (  0, 60,100),
    (  0,  0, 90),
    (  0,  0,110),
    (  0, 80,100),
    (  0,  0,230),
    (119, 11, 32),
    ( 64,128,128),
    (190,153,153),
    (150,120, 90),
    (153,153,153),
    (0,   64, 64),
    (0,  128,192),
    (128, 64,  0),
    (64,  64,128),
    (102,  0,  0),
    ( 51,  0, 51),
    ( 32, 32, 32),
    (  0,  0,142),
])

CLASS_COLORS_SYNTETIC: np.ndarray = np.asarray([
    [ 91, 255,  11], # brown ground
    [255, 255,   0], # yellow pipe
    [255,   0,   0], # red armature
    [ 77, 255, 216], # blue corner
    [ 65, 255,  80], # green adapter
    [128, 128, 128], # gray other
    ])

CLASS_COLORS: np.ndarray = CLASS_COLORS_KITTI if USE_KITTI_TRAIN_IDS else CLASS_COLORS_SYNTETIC


########################################################################
#                            Class Mappings                            #
########################################################################

THING_CLASSES: list[int] = [4, 5, 11, 13]  # deine gemappten Klassen die "things" sind
STUFF_CLASSES: list[int] = [i for i in range(SYNTHETIC_NUM_CLASSES) if i not in THING_CLASSES]