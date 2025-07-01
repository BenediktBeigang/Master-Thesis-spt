import numpy as np
import os.path as osp

USE_KITTI_TRAIN_IDS: bool = False


########################################################################
#                              Data splits                             #
########################################################################

TILES: dict[str, list[str]] = {
    'train': [
        '10_202506081336_frames_1_to_1056_noise_parts_processed',
        '11_202506081407_frames_1_to_1056_noise_parts_processed',
        '13_202506081442_frames_1_to_1056_noise_parts_processed',
        '14_202506130743_frames_1_to_1056_noise_parts_processed',
        '15_202506130819_frames_1_to_1056_noise_parts_processed',
        '16_202506130842_frames_1_to_1056_noise_parts_processed',
        '17_202506130906_frames_1_to_1056_noise_parts_processed',
        '18_202506130931_frames_1_to_1056_noise_parts_processed',
        '19_202506131118_frames_1_to_1056_noise_parts_processed',
        '20_202506131140_frames_1_to_1056_noise_parts_processed',
        '21_202506131216_frames_1_to_1056_noise_parts_processed',
        '22_202506131253_frames_1_to_1056_noise_parts_processed',
        '23_202506230623_frames_1_to_1056_noise_parts_processed',
        '24_202506230655_frames_1_to_1056_noise_parts_processed',
        '25_202506230727_frames_1_to_1056_noise_parts_processed',
        '26_202506230751_frames_1_to_1056_noise_parts_processed',
        '27_202506230820_frames_1_to_1056_noise_parts_processed',
        '28_202506230935_frames_1_to_1056_noise_parts_processed',
        '29_202506231006_frames_1_to_1056_noise_parts_processed',
        '30_202506210952_frames_1_to_1056_noise_parts_processed',
        '31_202506240609_frames_1_to_1056_noise_parts_processed',
        '32_202506240636_frames_1_to_1056_noise_parts_processed',
        '33_202506240705_frames_1_to_1056_noise_parts_processed',
        '34_202506240733_frames_1_to_1056_noise_parts_processed',
    ],
    
    'val': [
        '35_202506240804_frames_1_to_1056_noise_parts_processed',
        '36_202506240835_frames_1_to_1056_noise_parts_processed',
        '37_202506240910_frames_1_to_1056_noise_parts_processed',
        '38_202506240940_frames_1_to_1056_noise_parts_processed',
        '39_202506241010_frames_1_to_1056_noise_parts_processed',
        '40_202506241038_frames_1_to_1056_noise_parts_processed',
        '41_202506241135_frames_1_to_1056_noise_parts_processed',
    ],
    
    'test': [
        '42_202506241210_frames_1_to_1056_noise_parts_processed',
        '43_202506021352_frames_1_to_1049_noise_parts_processed',
        '44_202506241317_frames_1_to_1056_noise_parts_processed',
        '45_202506021650_frames_1_to_1049_noise_parts_processed',
    ]
}


########################################################################
#                                Labels                                #
########################################################################

SYNTHETIC_NUM_CLASSES: int = 15 if USE_KITTI_TRAIN_IDS else 5

# targeting kitti kitti-trainIDs [terrain, fence, pole, car, motorcycle]
ID2TRAINID: np.ndarray = np.asarray([9, 4, 5, 11, 13, 8])

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