import numpy as np
import os.path as osp


########################################################################
#                              Data splits                             #
########################################################################

TILES = {
    'train': [
        'seed_1/tile_0000',
        'seed_1/tile_0001',
        'seed_1/tile_0002',
        'seed_1/tile_0003',
        'seed_1/tile_0004',
        'seed_1/tile_0005',
        'seed_1/tile_0006',
        'seed_1/tile_0007',
    ],

    'val': [
        'seed_2/tile_0000',
        'seed_2/tile_0001',
        'seed_2/tile_0002',
        'seed_2/tile_0003',
        'seed_2/tile_0004',
        'seed_2/tile_0005',
        'seed_2/tile_0006',
        'seed_2/tile_0007',
    ],

    'test': [
        'seed_3/tile_0000',
        'seed_3/tile_0001',
        'seed_3/tile_0002',
        'seed_3/tile_0003',
        'seed_3/tile_0004',
        'seed_3/tile_0005',
        'seed_3/tile_0006',
        'seed_3/tile_0007',
    ]}


########################################################################
#                                Labels                                #
########################################################################

SYNTHETIC_NUM_CLASSES = 5

# targeting kitti kitti-trainIDs [terrain, fence, pole, car, motorcycle]
ID2TRAINID = np.asarray([9, 4, 5, 11, 13, 8])

CLASS_NAMES = [
    'ground',
    'pipe',
    'armature',
    'corner',
    'adapter',
    'other']

CLASS_COLORS = np.asarray([
    [ 91, 255,  11], # brown ground
    [255, 255,   0], # yellow pipe
    [255,   0,   0], # red armature
    [ 77, 255, 216], # blue corner
    [ 65, 255,  80], # green adapter
    [128, 128, 128], # gray other
    ])

THING_CLASSES = [5, 11, 13]
STUFF_CLASSES = [0, 1, 5]
