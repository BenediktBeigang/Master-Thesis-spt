import numpy as np
import os.path as osp


########################################################################
#                              Data splits                             #
########################################################################

TILES = {
    'train': [
        '10_202506081336',
    ],

    'val': [
        '11_202506081407',
    ],

    'test': [
        '13_202506081442',
    ]}


########################################################################
#                                Labels                                #
########################################################################

SYNTHETIC_NUM_CLASSES = 15

# targeting kitti kitti-trainIDs [terrain, fence, pole, car, motorcycle]
ID2TRAINID = np.asarray([9, 4, 5, 11, 13, 8])

# CLASS_NAMES = [
#     'ground',
#     'pipe',
#     'armature',
#     'corner',
#     'adapter',
#     'other']

# Erweitere CLASS_NAMES für alle 15 KITTI-360 Klassen
CLASS_NAMES = [
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

CLASS_COLORS = np.asarray([
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

# CLASS_COLORS = np.asarray([
#     [ 91, 255,  11], # brown ground
#     [255, 255,   0], # yellow pipe
#     [255,   0,   0], # red armature
#     [ 77, 255, 216], # blue corner
#     [ 65, 255,  80], # green adapter
#     [128, 128, 128], # gray other
#     ])

# THING_CLASSES = [5, 11, 13]
# STUFF_CLASSES = [0, 1, 5]

THING_CLASSES = [4, 5, 11, 13]  # deine gemappten Klassen die "things" sind
STUFF_CLASSES = [i for i in range(SYNTHETIC_NUM_CLASSES) if i not in THING_CLASSES]