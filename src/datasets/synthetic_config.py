from __future__ import annotations
import numpy as np
import os.path as osp

# USE_KITTI_TRAIN_IDS: bool = False
# ONLY_PIPES: bool = True


########################################################################
#                              Data splits                             #
########################################################################

TILES: dict[str, list[str]] = {
    'train': [
        # pipe-colors from dataset generation
        '0200_202507230825_frames_1_to_1059_noise_parts',
        '0201_202507230837_frames_1_to_1059_noise_parts',
        '0202_202507230848_frames_1_to_1059_noise_parts',
        '0203_202507230900_frames_1_to_1059_noise_parts',
        '0204_202507230912_frames_1_to_1059_noise_parts',
        '0205_202507230924_frames_1_to_1059_noise_parts',
        '0206_202507230936_frames_1_to_1059_noise_parts',
        '0207_202507230949_frames_1_to_1059_noise_parts',
        '0208_202507231000_frames_1_to_1059_noise_parts',
        '0209_202507231012_frames_1_to_1059_noise_parts',
        '0210_202507231025_frames_1_to_1059_noise_parts',
        '0211_202507231036_frames_1_to_1059_noise_parts',
        '0212_202507231047_frames_1_to_1059_noise_parts',
        '0213_202507231059_frames_1_to_1059_noise_parts',
        '0214_202507231111_frames_1_to_1059_noise_parts',
        '0215_202507231236_frames_1_to_1059_noise_parts',
        '0216_202507231247_frames_1_to_1059_noise_parts',
        '0217_202507231259_frames_1_to_1059_noise_parts',
        '0218_202507231311_frames_1_to_1059_noise_parts',
        '0219_202507231322_frames_1_to_1059_noise_parts',
        '0220_202507231334_frames_1_to_1059_noise_parts',
        '0221_202507231345_frames_1_to_1059_noise_parts',
        '0222_202507231357_frames_1_to_1059_noise_parts',
        '0223_202507231410_frames_1_to_1059_noise_parts',
        '0224_202507231422_frames_1_to_1059_noise_parts',
        '0225_202507231434_frames_1_to_1059_noise_parts',
        
        # pipe-colors have random value based on their original color
        '0226_202507231445_frames_1_to_1059_noise_parts',
        '0227_202507231457_frames_1_to_1059_noise_parts',
        '0228_202507231509_frames_1_to_1059_noise_parts',
        '0229_202507231521_frames_1_to_1059_noise_parts',
        '0230_202507231533_frames_1_to_1059_noise_parts',
        '0231_202507231545_frames_1_to_1059_noise_parts',
        '0232_202507231557_frames_1_to_1059_noise_parts',
        '0233_202507231608_frames_1_to_1059_noise_parts',
        '0234_202507231620_frames_1_to_1059_noise_parts',
        '0235_202507231632_frames_1_to_1059_noise_parts',
        '0236_202507231643_frames_1_to_1059_noise_parts',
        '0237_202507231655_frames_1_to_1059_noise_parts',
        '0238_202507231707_frames_1_to_1059_noise_parts',
        '0239_202507260905_frames_1_to_1059_noise_parts',
        '0240_202507260918_frames_1_to_1059_noise_parts',
        '0241_202507260931_frames_1_to_1059_noise_parts',
        '0242_202507260943_frames_1_to_1059_noise_parts',
        '0243_202507260956_frames_1_to_1059_noise_parts',
        '0244_202507261008_frames_1_to_1059_noise_parts',
        '0245_202507261020_frames_1_to_1059_noise_parts',
        '0246_202507261033_frames_1_to_1059_noise_parts',
        '0247_202507261046_frames_1_to_1059_noise_parts',
        '0248_202507261058_frames_1_to_1059_noise_parts',
        '0249_202507261110_frames_1_to_1059_noise_parts',
        '0250_202507261123_frames_1_to_1059_noise_parts',
        '0251_202507261134_frames_1_to_1059_noise_parts',
        
        # pipe-colors have one random value for the whole cloud
        '0252_202507261146_frames_1_to_1059_noise_parts',
        '0253_202507261159_frames_1_to_1059_noise_parts',
        '0254_202507261210_frames_1_to_1059_noise_parts',
        '0255_202507261241_frames_1_to_1059_noise_parts',
        '0256_202507261253_frames_1_to_1059_noise_parts',
        '0257_202507261304_frames_1_to_1059_noise_parts',
        '0258_202507261316_frames_1_to_1059_noise_parts',
        '0259_202507261329_frames_1_to_1059_noise_parts',
        '0260_202507261627_frames_1_to_1059_noise_parts',
        '0261_202507261639_frames_1_to_1059_noise_parts',
        '0262_202507261651_frames_1_to_1059_noise_parts',
        '0263_202507261702_frames_1_to_1059_noise_parts',
        '0264_202507261714_frames_1_to_1059_noise_parts',
        '0265_202507261725_frames_1_to_1059_noise_parts',
        '0266_202507261737_frames_1_to_1059_noise_parts',
        '0267_202507261749_frames_1_to_1059_noise_parts',
        '0268_202507261801_frames_1_to_1059_noise_parts',
        '0269_202507261813_frames_1_to_1059_noise_parts',
        '0270_202507261824_frames_1_to_1059_noise_parts',
        '0271_202507261836_frames_1_to_1059_noise_parts',
        '0272_202507261848_frames_1_to_1059_noise_parts',
        '0273_202507261900_frames_1_to_1059_noise_parts',
        '0274_202507261912_frames_1_to_1059_noise_parts',
        '0275_202507261924_frames_1_to_1059_noise_parts',
        '0276_202507261935_frames_1_to_1059_noise_parts',
        '0277_202507261948_frames_1_to_1059_noise_parts',
        '0278_202507262000_frames_1_to_1059_noise_parts',
        '0279_202507262012_frames_1_to_1059_noise_parts',
    ],
    
    'val': [
        # pipe-colors from dataset generation
        '0280_202507262025_frames_1_to_1059_noise_parts',
        '0281_202507262037_frames_1_to_1059_noise_parts',
        '0282_202507262049_frames_1_to_1059_noise_parts',
        '0283_202507262101_frames_1_to_1059_noise_parts',
        '0284_202507262113_frames_1_to_1059_noise_parts',
        '0285_202507262125_frames_1_to_1059_noise_parts',

        # pipe-colors have random value based on their original color
        '0286_202507262137_frames_1_to_1059_noise_parts',
        '0287_202507262149_frames_1_to_1059_noise_parts',
        '0288_202507262201_frames_1_to_1059_noise_parts',
        '0289_202507262213_frames_1_to_1059_noise_parts',
        '0290_202507262225_frames_1_to_1059_noise_parts',
        '0291_202507262236_frames_1_to_1059_noise_parts',

        # pipe-colors have one random value for the whole cloud
        '0292_202507262248_frames_1_to_1059_noise_parts',
        '0293_202507262300_frames_1_to_1059_noise_parts',
        '0294_202507262312_frames_1_to_1059_noise_parts',
        '0295_202507262323_frames_1_to_1059_noise_parts',
        '0296_202507262336_frames_1_to_1059_noise_parts',
        '0297_202507262348_frames_1_to_1059_noise_parts',
        '0298_202507262359_frames_1_to_1059_noise_parts',
        '0299_202507270011_frames_1_to_1059_noise_parts',
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

# SYNTHETIC_NUM_CLASSES: int = 15 if USE_KITTI_TRAIN_IDS else (3 if ONLY_PIPES else 5)
SYNTHETIC_NUM_CLASSES: int = 4

def remapArrays() -> np.ndarray | None:
    # if not USE_KITTI_TRAIN_IDS and not ONLY_PIPES:
    # return np.asarray([0, 1, 2, 3, 4, 0, 1, 5, 5])
    
    # if not USE_KITTI_TRAIN_IDS and ONLY_PIPES:
    return np.asarray([0, 1, 2, 1, 1, 0, 1, 3, 3])
    
    # if USE_KITTI_TRAIN_IDS and not ONLY_PIPES:
    # targeting kitti kitti-trainIDs [terrain/9, fence/4, pole/5, car/11, motorcycle/13], is only used if USE_KITTI_TRAIN_IDS is True
    # return np.asarray([9, 4, 5, 11, 13, 8, 8, 8, 8])
    
    # if USE_KITTI_TRAIN_IDS and ONLY_PIPES:
    # return np.asarray([9, 4, 4, 4, 4, 8, 8, 8, 8])

ID2TRAINID: np.ndarray | None =  remapArrays()

############################
####### CLASS NAMES ########
############################

# CLASS_NAMES_SYNTHETIC_DATASET: list[str] = [
#     'ground',           # 0
#     'pipe',             # 1
#     'armature',         # 2
#     'corner',           # 3
#     'adapter',          # 4
#     'feet',             # 5
#     'tpiece',           # 6
#     'building',         # 7
#     'armatureDecoy',    # 8
#     'other'             # 9
# ]

CLASS_NAMES_SYNTHETIC: list[str] = [
    'ground',     # 0
    'pipe',       # 1
    'armature',   # 2
    'corner',     # 3
    'adapter',    # 4
    'other'       # 5
]

CLASS_NAMES_SYNTHETIC_PIPE_ONLY: list[str] = [
    'ground',     # 0
    'pipe',       # 1
    'armature',   # 2
    'background', # 3
    'other'       # 4
]

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

# CLASS_NAMES: list[str] = CLASS_NAMES_KITTI if USE_KITTI_TRAIN_IDS else CLASS_NAMES_SYNTHETIC
CLASS_NAMES: list[str] = CLASS_NAMES_SYNTHETIC_PIPE_ONLY

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

CLASS_COLORS_SYNTETIC_PIPE_ONLY: np.ndarray = np.asarray([
    [ 91, 255,  11], # brown ground
    [255, 255,   0], # yellow pipe
    [255,   0,   0], # red armature
    [128, 128, 128], # background corner
    [128, 128, 128], # gray other
    ])

CLASS_COLORS_SYNTETIC: np.ndarray = np.asarray([
    [ 91, 255,  11], # brown ground
    [255, 255,   0], # yellow pipe
    [255,   0,   0], # red armature
    [ 77, 255, 216], # blue corner
    [ 65, 255,  80], # green adapter
    [128, 128, 128], # gray feet
    [128, 128, 128], # gray tpiece
    [128, 128, 128], # gray building
    [128, 128, 128], # gray armatureDecoy
    [128, 128, 128], # gray other
    ])

CLASS_COLORS: np.ndarray = CLASS_COLORS_SYNTETIC_PIPE_ONLY # CLASS_COLORS_KITTI if USE_KITTI_TRAIN_IDS else CLASS_COLORS_SYNTETIC



########################################################################
#                            Class Mappings                            #
########################################################################

# THING_CLASSES: list[int] = [4, 5, 11, 13]  # deine gemappten Klassen die "things" sind
# STUFF_CLASSES: list[int] = [i for i in range(SYNTHETIC_NUM_CLASSES) if i not in THING_CLASSES]