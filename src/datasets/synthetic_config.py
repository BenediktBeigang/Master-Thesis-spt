from __future__ import annotations
import numpy as np
import os.path as osp

# USE_KITTI_TRAIN_IDS: bool = False
# ONLY_PIPES: bool = True


########################################################################
#                              Data splits                             #
########################################################################

TILES: dict[str, list[str]] = {
    "train": [
        "700_202509022301_frames_1_to_1059_noise_parts",
        "701_202509022322_frames_1_to_849_noise_parts",
        "702_202509021049_frames_1_to_1059_noise_parts",
        "703_202509030002_frames_1_to_219_noise_parts",
        "704_202509030006_frames_1_to_1059_noise_parts",
        "705_202509030027_frames_1_to_1059_noise_parts",
        "706_202509030048_frames_1_to_1059_noise_parts",
        "707_202509021132_frames_1_to_639_noise_parts",
        "708_202509021139_frames_1_to_1059_noise_parts",
        "709_202509021150_frames_1_to_1059_noise_parts",
        "710_202509021200_frames_1_to_1059_noise_parts",
        "711_202509021210_frames_1_to_1059_noise_parts",
        "712_202509021220_frames_1_to_1059_noise_parts",
        "713_202509021230_frames_1_to_1059_noise_parts",
        "714_202509021240_frames_1_to_1059_noise_parts",
        "715_202509021251_frames_1_to_1059_noise_parts",
        "716_202509021301_frames_1_to_639_noise_parts",
        "717_202509021309_frames_1_to_1059_noise_parts",
        "718_202509021319_frames_1_to_429_noise_parts",
        "719_202509021324_frames_1_to_1059_noise_parts",
        "720_202509021334_frames_1_to_1059_noise_parts",
        "721_202509021345_frames_1_to_1059_noise_parts",
        "722_202509021355_frames_1_to_1059_noise_parts",
        "723_202509021406_frames_1_to_1059_noise_parts",
        "724_202509021416_frames_1_to_1059_noise_parts",
        "725_202509021425_frames_1_to_1059_noise_parts",
        "726_202509021435_frames_1_to_1059_noise_parts",
        "727_202509021446_frames_1_to_1059_noise_parts",
        "728_202509021456_frames_1_to_1059_noise_parts",
        "729_202509021506_frames_1_to_1059_noise_parts",
        "730_202509021515_frames_1_to_1059_noise_parts",
        "731_202509021526_frames_1_to_1059_noise_parts",
        "732_202509021537_frames_1_to_1059_noise_parts",
        "733_202509021547_frames_1_to_849_noise_parts",
        "734_202509021555_frames_1_to_429_noise_parts",
        "735_202509021559_frames_1_to_1059_noise_parts",
        "736_202509021609_frames_1_to_1059_noise_parts",
        "737_202509021619_frames_1_to_1059_noise_parts",
        "738_202509021629_frames_1_to_1059_noise_parts",
        "739_202509021640_frames_1_to_1059_noise_parts",
        "740_202509021650_frames_1_to_1059_noise_parts",
        "741_202509021701_frames_1_to_1059_noise_parts",
        "742_202509021711_frames_1_to_1059_noise_parts",
        "743_202509021722_frames_1_to_219_noise_parts",
        "744_202509021724_frames_1_to_1059_noise_parts",
        "745_202509021735_frames_1_to_1059_noise_parts",
        "746_202509021745_frames_1_to_639_noise_parts",
        "747_202509021752_frames_1_to_1059_noise_parts",
        "748_202509021803_frames_1_to_849_noise_parts",
        "749_202509021811_frames_1_to_1059_noise_parts",
        "750_202509021821_frames_1_to_1059_noise_parts",
        "751_202509021831_frames_1_to_429_noise_parts",
        "752_202509021835_frames_1_to_219_noise_parts",
        "753_202509021837_frames_1_to_1059_noise_parts",
        "754_202509021848_frames_1_to_1059_noise_parts",
        "755_202509021858_frames_1_to_1059_noise_parts",
        "756_202509021908_frames_1_to_1059_noise_parts",
        "757_202509021918_frames_1_to_1059_noise_parts",
        "758_202509021929_frames_1_to_1059_noise_parts",
        "759_202509021941_frames_1_to_1059_noise_parts",
        "760_202509021952_frames_1_to_1059_noise_parts",
        "761_202509022003_frames_1_to_1059_noise_parts",
        "762_202509022013_frames_1_to_1059_noise_parts",
        "763_202509022023_frames_1_to_1059_noise_parts",
        "764_202509022034_frames_1_to_1059_noise_parts",
        "765_202509022045_frames_1_to_1059_noise_parts",
        "766_202509022056_frames_1_to_1059_noise_parts",
        "767_202509022106_frames_1_to_1059_noise_parts",
        "768_202509022117_frames_1_to_1059_noise_parts",
        "769_202509022128_frames_1_to_1059_noise_parts",
        "770_202509022139_frames_1_to_1059_noise_parts",
        "771_202509022149_frames_1_to_1059_noise_parts",
        "772_202509022200_frames_1_to_1059_noise_parts",
        "773_202509022211_frames_1_to_1059_noise_parts",
        "774_202509022221_frames_1_to_1059_noise_parts",
        "775_202509022231_frames_1_to_1059_noise_parts",
        "776_202509022241_frames_1_to_1059_noise_parts",
        "777_202509030852_frames_1_to_429_noise_parts",
        "778_202509030904_frames_1_to_1059_noise_parts",
        "779_202509030927_frames_1_to_219_noise_parts",
    ],
    "val": [
        "780_202509030932_frames_1_to_1059_noise_parts",
        "781_202509031112_frames_1_to_1059_noise_parts",
        "782_202509031133_frames_1_to_1059_noise_parts",
        "783_202509031157_frames_1_to_1059_noise_parts",
        "784_202509031218_frames_1_to_639_noise_parts",
        "785_202509031230_frames_1_to_1059_noise_parts",
        "786_202509031251_frames_1_to_1059_noise_parts",
        "787_202509031312_frames_1_to_1059_noise_parts",
        "788_202509031333_frames_1_to_219_noise_parts",
        "789_202509031337_frames_1_to_849_noise_parts",
        "790_202509031356_frames_1_to_1059_noise_parts",
        "791_202509031419_frames_1_to_1059_noise_parts",
        "792_202509031439_frames_1_to_429_noise_parts",
        "793_202509031449_frames_1_to_219_noise_parts",
        "794_202509031454_frames_1_to_1059_noise_parts",
        "795_202509031514_frames_1_to_1059_noise_parts",
        "796_202509031533_frames_1_to_1059_noise_parts",
        "797_202509031553_frames_1_to_1059_noise_parts",
        "798_202509031614_frames_1_to_1059_noise_parts",
        "799_202509040659_frames_1_to_429_noise_parts",
    ],
    "test": [],
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

# SYNTHETIC_NUM_CLASSES: int = 15 # KITTI-360
SYNTHETIC_NUM_CLASSES: int = 4  # pipes, armature, ground, background
# SYNTHETIC_NUM_CLASSES: int = 5 # pipes, armature, corner, adapter, ground


def remapArray() -> np.ndarray | None:
    """Remap the original labels to training labels.
    Returns:
        A numpy array of shape (N,) where N is the max label + 1.
        The value at each index is the new label for that index.
        If no remapping is needed, return None.
    """
    # if not USE_KITTI_TRAIN_IDS and not ONLY_PIPES:
    # return np.asarray([0, 1, 2, 3, 4, 0, 1, 5, 5]) # 0: ground, 1: pipe, 2: armature, 3: corner, 4: adapter, 5: other

    return np.asarray(
        [0, 1, 2, 1, 1, 3, 1, 3, 3]
    )  # 0: ground, 1: pipe, 2: armature, 3: background

    # targeting kitti kitti-trainIDs [terrain/9, fence/4, pole/5, car/11, motorcycle/13], is only used if USE_KITTI_TRAIN_IDS is True
    # return np.asarray([9, 4, 5, 11, 13, 8, 8, 8, 8]) # KITTI-360

    # return np.asarray([9, 4, 4, 4, 4, 8, 8, 8, 8]) # KITTI-360 but only armature, corner and adapter are pipes


ID2TRAINID: np.ndarray | None = remapArray()

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
    "ground",  # 0
    "pipe",  # 1
    "armature",  # 2
    "corner",  # 3
    "adapter",  # 4
    "other",  # 5
]

CLASS_NAMES_SYNTHETIC_PIPE_ONLY: list[str] = [
    "ground",  # 0
    "pipe",  # 1
    "armature",  # 2
    "background",  # 3
    "other",  # 4
]

# Erweitere CLASS_NAMES für alle 15 KITTI-360 Klassen
CLASS_NAMES_KITTI: list[str] = [
    "road",  # 0
    "sidewalk",  # 1
    "building",  # 2
    "wall",  # 3
    "fence",  # 4 -> deine 'pipe' Klasse
    "pole",  # 5 -> deine 'armature' Klasse
    "traffic light",  # 6
    "traffic sign",  # 7
    "vegetation",  # 8
    "terrain",  # 9 -> deine 'ground' Klasse
    "sky",  # 10
    "car",  # 11 -> deine 'corner' Klasse
    "truck",  # 12
    "motorcycle",  # 13 -> deine 'adapter' Klasse
    "other",  # 14
    "ignored",
]  # 15 -> void/ignored Klasse

# CLASS_NAMES: list[str] = CLASS_NAMES_KITTI if USE_KITTI_TRAIN_IDS else CLASS_NAMES_SYNTHETIC
CLASS_NAMES: list[str] = CLASS_NAMES_SYNTHETIC_PIPE_ONLY

#############################
####### CLASS COLORS ########
#############################

CLASS_COLORS_KITTI: np.ndarray = np.asarray(
    [
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (111, 74, 0),
        (81, 0, 81),
        (128, 64, 128),
        (244, 35, 232),
        (250, 170, 160),
        (230, 150, 140),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (180, 165, 180),
        (150, 100, 100),
        (150, 120, 90),
        (153, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 0, 90),
        (0, 0, 110),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
        (64, 128, 128),
        (190, 153, 153),
        (150, 120, 90),
        (153, 153, 153),
        (0, 64, 64),
        (0, 128, 192),
        (128, 64, 0),
        (64, 64, 128),
        (102, 0, 0),
        (51, 0, 51),
        (32, 32, 32),
        (0, 0, 142),
    ]
)

CLASS_COLORS_SYNTETIC_PIPE_ONLY: np.ndarray = np.asarray(
    [
        [91, 255, 11],  # brown ground
        [0, 255, 0],  # green pipe
        [255, 128, 0],  # orange armature
        [128, 128, 128],  # gray background
        [128, 128, 128],  # gray other
    ]
)

CLASS_COLORS_SYNTETIC: np.ndarray = np.asarray(
    [
        [91, 255, 11],  # brown ground
        [255, 255, 0],  # yellow pipe
        [255, 0, 0],  # red armature
        [77, 255, 216],  # blue corner
        [65, 255, 80],  # green adapter
        [128, 128, 128],  # gray feet
        [128, 128, 128],  # gray tpiece
        [128, 128, 128],  # gray building
        [128, 128, 128],  # gray armatureDecoy
        [128, 128, 128],  # gray other
    ]
)

CLASS_COLORS: np.ndarray = (
    CLASS_COLORS_SYNTETIC_PIPE_ONLY  # CLASS_COLORS_KITTI if USE_KITTI_TRAIN_IDS else CLASS_COLORS_SYNTETIC
)


########################################################################
#                            Class Mappings                            #
########################################################################

# THING_CLASSES: list[int] = [4, 5, 11, 13]  # deine gemappten Klassen die "things" sind
# STUFF_CLASSES: list[int] = [i for i in range(SYNTHETIC_NUM_CLASSES) if i not in THING_CLASSES]
