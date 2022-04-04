import os
import os.path as osp
from easydict import EasyDict as edict
import math
import numpy as np

__C = edict()
# Consumers can get config by:
#    import config as cfg
cfg = __C

# for dataset dir
__C.TRAIN_DIR = 'kitti/training'
__C.VAL_DIR = 'kitti/validation'

# for gpu allocation
__C.GPU_AVAILABLE = '0,1'
__C.GPU_USE_COUNT = len(__C.GPU_AVAILABLE.split(','))
__C.GPU_MEMORY_FRACTION = 1

# selected object
__C.CLASSES = ['Car', 'Truck', 'Van', 'Tram', 'Pedestrian', 'Cyclist']

__C.CLASSES_AVG = {'Cyclist': np.array([1.73532436, 0.58028152, 1.77413709]),
                   'Van': np.array([2.18928571, 1.90979592, 5.07087755]),
                   'Tram': np.array([3.56092896, 2.39601093, 18.34125683]),
                   'Car': np.array([1.52159147, 1.64443089, 3.85813679]),
                   'Pedestrian': np.array([1.75554637, 0.66860882, 0.87623049]),
                   'Truck': np.array([3.07392252, 2.63079903, 11.2190799])}

# Root directory of project
__C.CHECKPOINT_DIR = osp.join('checkpoint')
__C.LOG_DIR = osp.join('log')

# for data preprocessing
# images

__C.IMAGE_WIDTH = 224
__C.IMAGE_HEIGHT = 224
__C.IMAGE_CHANNEL = 3
__C.BIN = 2
__C.OVERLAP = 0.1

# for rpn nms
__C.RPN_NMS_POST_TOPK = 20
__C.RPN_NMS_THRESH = 0.1
__C.RPN_SCORE_THRESH = 0.96

# utils
__C.CORNER2CENTER_AVG = True  # average version or max version

if __name__ == '__main__':
    print('__C.ROOT_DIR = ' + __C.IMAGE_WIDTH)
    #print('__C.DATA_SETS_DIR = ' + __C.DATA_SETS_DIR)
