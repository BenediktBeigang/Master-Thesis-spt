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
        # "300_202508141424_frames_1_to_1059_noise_parts",
        # "301_202508141447_frames_1_to_1059_noise_parts",
        # "302_202508141501_frames_1_to_1059_noise_parts",
        # "303_202508141514_frames_1_to_1059_noise_parts",
        # "304_202508141527_frames_1_to_1059_noise_parts",
        # "305_202508141540_frames_1_to_1059_noise_parts",
        # "306_202508141553_frames_1_to_1059_noise_parts",
        # "307_202508141606_frames_1_to_1059_noise_parts",
        # "308_202508141620_frames_1_to_1059_noise_parts",
        # "309_202508141632_frames_1_to_1059_noise_parts",
        # "310_202508141645_frames_1_to_1059_noise_parts",
        # "311_202508141658_frames_1_to_1059_noise_parts",
        # "312_202508141710_frames_1_to_1059_noise_parts",
        # "313_202508141723_frames_1_to_1059_noise_parts",
        # "314_202508141736_frames_1_to_1059_noise_parts",
        # "315_202508141750_frames_1_to_1059_noise_parts",
        # "316_202508141804_frames_1_to_1059_noise_parts",
        # "317_202508141816_frames_1_to_1059_noise_parts",
        # "318_202508141829_frames_1_to_1059_noise_parts",
        # "319_202508141842_frames_1_to_1059_noise_parts",
        # "320_202508141854_frames_1_to_1059_noise_parts",
        # "321_202508141909_frames_1_to_1059_noise_parts",
        # "322_202508141922_frames_1_to_1059_noise_parts",
        # "323_202508141936_frames_1_to_1059_noise_parts",
        # "324_202508141949_frames_1_to_1059_noise_parts",
        # "325_202508142004_frames_1_to_1059_noise_parts",
        # "326_202508142017_frames_1_to_1059_noise_parts",
        # "327_202508142030_frames_1_to_1059_noise_parts",
        # "328_202508142042_frames_1_to_1059_noise_parts",
        # "329_202508142056_frames_1_to_1059_noise_parts",
        # "330_202508142109_frames_1_to_1059_noise_parts",
        # "331_202508142121_frames_1_to_1059_noise_parts",
        # "332_202508142134_frames_1_to_1059_noise_parts",
        # "333_202508142146_frames_1_to_1059_noise_parts",
        # "334_202508142159_frames_1_to_1059_noise_parts",
        # "335_202508142211_frames_1_to_1059_noise_parts",
        # "336_202508142224_frames_1_to_1059_noise_parts",
        # "337_202508142237_frames_1_to_1059_noise_parts",
        # "338_202508142251_frames_1_to_1059_noise_parts",
        # "339_202508142305_frames_1_to_1059_noise_parts",
        # "340_202508142320_frames_1_to_1059_noise_parts",
        # "341_202508142333_frames_1_to_1059_noise_parts",
        # "342_202508142345_frames_1_to_1059_noise_parts",
        # "343_202508150000_frames_1_to_1059_noise_parts",
        # "344_202508150013_frames_1_to_1059_noise_parts",
        # "345_202508150026_frames_1_to_1059_noise_parts",
        # "346_202508150039_frames_1_to_1059_noise_parts",
        # "347_202508150052_frames_1_to_1059_noise_parts",
        # "348_202508150105_frames_1_to_1059_noise_parts",
        # "349_202508150117_frames_1_to_1059_noise_parts",
        # "350_202508150129_frames_1_to_1059_noise_parts",
        # "351_202508150142_frames_1_to_1059_noise_parts",
        # "352_202508150154_frames_1_to_1059_noise_parts",
        # "353_202508150208_frames_1_to_1059_noise_parts",
        # "354_202508150220_frames_1_to_1059_noise_parts",
        # "355_202508150233_frames_1_to_1059_noise_parts",
        # "356_202508150246_frames_1_to_1059_noise_parts",
        # "357_202508150259_frames_1_to_1059_noise_parts",
        # "358_202508150312_frames_1_to_1059_noise_parts",
        # "359_202508150328_frames_1_to_1059_noise_parts",
        # "360_202508150340_frames_1_to_1059_noise_parts",
        # "361_202508150353_frames_1_to_1059_noise_parts",
        # "362_202508150405_frames_1_to_1059_noise_parts",
        # "363_202508150420_frames_1_to_1059_noise_parts",
        # "364_202508150434_frames_1_to_1059_noise_parts",
        # "365_202508150447_frames_1_to_1059_noise_parts",
        # "366_202508150459_frames_1_to_1059_noise_parts",
        # "367_202508150513_frames_1_to_1059_noise_parts",
        # "368_202508150526_frames_1_to_1059_noise_parts",
        # "369_202508150539_frames_1_to_1059_noise_parts",
        # "370_202508150552_frames_1_to_1059_noise_parts",
        # "371_202508150607_frames_1_to_1059_noise_parts",
        # "372_202508150619_frames_1_to_1059_noise_parts",
        # "373_202508171611_frames_1_to_1059_noise_parts",
        # "374_202508171623_frames_1_to_1059_noise_parts",
        # "375_202508171636_frames_1_to_1059_noise_parts",
        # "376_202508171649_frames_1_to_1059_noise_parts",
        # "377_202508171701_frames_1_to_1059_noise_parts",
        # "378_202508171900_frames_1_to_1059_noise_parts",
        # "379_202508171913_frames_1_to_1059_noise_parts",
        ################################################
        "500_202508251004_frames_1_to_429_noise_parts",
        "501_202508251010_frames_1_to_219_noise_parts",
        "502_202508251012_frames_1_to_639_noise_parts",
        "503_202508251020_frames_1_to_429_noise_parts",
        "504_202508251026_frames_1_to_1059_noise_parts",
        "505_202508251039_frames_1_to_429_noise_parts",
        "506_202508251043_frames_1_to_219_noise_parts",
        "507_202508251045_frames_1_to_429_noise_parts",
        "508_202508251051_frames_1_to_429_noise_parts",
        "509_202508251057_frames_1_to_849_noise_parts",
        "510_202508251107_frames_1_to_429_noise_parts",
        "511_202508251112_frames_1_to_429_noise_parts",
        "512_202508251116_frames_1_to_1059_noise_parts",
        "513_202508251129_frames_1_to_849_noise_parts",
        "514_202508251139_frames_1_to_849_noise_parts",
        "515_202508251147_frames_1_to_429_noise_parts",
        "516_202508251152_frames_1_to_429_noise_parts",
        "517_202508251157_frames_1_to_1059_noise_parts",
        "518_202508251209_frames_1_to_639_noise_parts",
        "519_202508251215_frames_1_to_639_noise_parts",
        "520_202508251302_frames_1_to_849_noise_parts",
        "521_202508251319_frames_1_to_219_noise_parts",
        "522_202508251323_frames_1_to_1059_noise_parts",
        "523_202508251342_frames_1_to_219_noise_parts",
        "524_202508251347_frames_1_to_219_noise_parts",
        "525_202508251352_frames_1_to_429_noise_parts",
        "526_202508251400_frames_1_to_849_noise_parts",
        "527_202508251416_frames_1_to_1059_noise_parts",
        "528_202508251437_frames_1_to_219_noise_parts",
        "529_202508251442_frames_1_to_219_noise_parts",
        "530_202508251447_frames_1_to_219_noise_parts",
        "531_202508251455_frames_1_to_1059_noise_parts",
        "532_202508251516_frames_1_to_639_noise_parts",
        "533_202508251530_frames_1_to_639_noise_parts",
        "534_202508251540_frames_1_to_639_noise_parts",
        "535_202508251552_frames_1_to_1059_noise_parts",
        "536_202508251612_frames_1_to_849_noise_parts",
        "537_202508251629_frames_1_to_849_noise_parts",
        "538_202508251642_frames_1_to_849_noise_parts",
        "539_202508251658_frames_1_to_849_noise_parts",
        "540_202508251711_frames_1_to_1059_noise_parts",
        "541_202508251730_frames_1_to_429_noise_parts",
        "542_202508251737_frames_1_to_849_noise_parts",
        "543_202508251752_frames_1_to_849_noise_parts",
        "544_202508251808_frames_1_to_639_noise_parts",
        "545_202508251819_frames_1_to_429_noise_parts",
        "546_202508251829_frames_1_to_639_noise_parts",
        "547_202508251841_frames_1_to_639_noise_parts",
        "548_202508251855_frames_1_to_219_noise_parts",
        "549_202508251859_frames_1_to_219_noise_parts",
        "550_202508251903_frames_1_to_1059_noise_parts",
        "551_202508251923_frames_1_to_849_noise_parts",
        "552_202508251939_frames_1_to_429_noise_parts",
        "553_202508251946_frames_1_to_429_noise_parts",
        "554_202508251956_frames_1_to_849_noise_parts",
        "555_202508252014_frames_1_to_1059_noise_parts",
        "556_202508252035_frames_1_to_849_noise_parts",
        "557_202508252052_frames_1_to_639_noise_parts",
        "558_202508252105_frames_1_to_1059_noise_parts",
        "559_202508252125_frames_1_to_849_noise_parts",
        "560_202508252142_frames_1_to_1059_noise_parts",
        "561_202508252202_frames_1_to_1059_noise_parts",
        "562_202508252223_frames_1_to_219_noise_parts",
        "563_202508252227_frames_1_to_639_noise_parts",
        "564_202508252240_frames_1_to_1059_noise_parts",
        "565_202508252300_frames_1_to_1059_noise_parts",
        "566_202508252319_frames_1_to_1059_noise_parts",
        "567_202508252341_frames_1_to_219_noise_parts",
        "568_202508252348_frames_1_to_219_noise_parts",
        "569_202508252352_frames_1_to_1059_noise_parts",
        "570_202508260013_frames_1_to_639_noise_parts",
        "571_202508260027_frames_1_to_429_noise_parts",
        "572_202508260034_frames_1_to_1059_noise_parts",
        "573_202508260053_frames_1_to_219_noise_parts",
        "574_202508260056_frames_1_to_1059_noise_parts",
        "575_202508260118_frames_1_to_849_noise_parts",
        "576_202508260132_frames_1_to_429_noise_parts",
        "577_202508260139_frames_1_to_429_noise_parts",
        "578_202508260146_frames_1_to_429_noise_parts",
        "579_202508260153_frames_1_to_429_noise_parts",
    ],
    "val": [
        # "380_202508171926_frames_1_to_1059_noise_parts",
        # "381_202508171939_frames_1_to_1059_noise_parts",
        # "382_202508171951_frames_1_to_1059_noise_parts",
        # "383_202508172004_frames_1_to_1059_noise_parts",
        # "384_202508172016_frames_1_to_1059_noise_parts",
        # "385_202508172029_frames_1_to_1059_noise_parts",
        # "386_202508172041_frames_1_to_1059_noise_parts",
        # "387_202508172054_frames_1_to_1059_noise_parts",
        # "388_202508172106_frames_1_to_1059_noise_parts",
        # "389_202508172120_frames_1_to_1059_noise_parts",
        # "390_202508172131_frames_1_to_1059_noise_parts",
        # "391_202508172146_frames_1_to_1059_noise_parts",
        # "392_202508172159_frames_1_to_1059_noise_parts",
        # "393_202508172212_frames_1_to_1059_noise_parts",
        # "394_202508172224_frames_1_to_1059_noise_parts",
        # "395_202508172237_frames_1_to_1059_noise_parts",
        # "396_202508172249_frames_1_to_1059_noise_parts",
        # "397_202508172301_frames_1_to_1059_noise_parts",
        # "398_202508172315_frames_1_to_1059_noise_parts",
        # "399_202508172328_frames_1_to_1059_noise_parts",
        ################################################
        "580_202508260200_frames_1_to_429_noise_parts",
        "581_202508260210_frames_1_to_639_noise_parts",
        "582_202508260225_frames_1_to_219_noise_parts",
        "583_202508260945_frames_1_to_1059_noise_parts",
        "584_202508260608_frames_1_to_429_noise_parts",
        "585_202508260615_frames_1_to_219_noise_parts",
        "586_202508260619_frames_1_to_219_noise_parts",
        "587_202508260623_frames_1_to_849_noise_parts",
        "588_202508260640_frames_1_to_219_noise_parts",
        "589_202508260644_frames_1_to_849_noise_parts",
        "590_202508260703_frames_1_to_429_noise_parts",
        "591_202508260713_frames_1_to_1059_noise_parts",
        "592_202508260734_frames_1_to_1059_noise_parts",
        "593_202508260754_frames_1_to_429_noise_parts",
        "594_202508260802_frames_1_to_849_noise_parts",
        "595_202508260820_frames_1_to_429_noise_parts",
        "596_202508260827_frames_1_to_639_noise_parts",
        "597_202508260841_frames_1_to_1059_noise_parts",
        "598_202508260902_frames_1_to_1059_noise_parts",
        "599_202508260923_frames_1_to_1059_noise_parts",
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
