# -*- coding: utf-8 -*-
# @Time    : 2021-09-29 16:30
# @Author  : zhangbowen


BATCH_SIZE = 4
EPOCH = 50
VAL_DATA = 0.2

INPUT_SIZE = 475 # PSPNet
# INPUT_SIZE = 475 # DeepLabv3
# INPUT_SIZE = 480 # UNet

NAME_NET = 'pspnet' 
# NAME_NET = 'deeplabv3' 
# NAME_NET = 'unet'


# customize
NUM_CLASSES = 6 
NAME_LABEL = {0:'background', 1:'obstacle', 2: 'road',
              3:'sidewalk', 4:'crosswalk', 5:'yellow-warning-block'
}


# # cityscapes
# NUM_CLASSES = 34
# NAME_LABEL = {0:'unlabeled', 1:'ego vehicle', 2: 'rectification border', 3:'out of roi', 4:'static',
#           5:'dynamic', 6:'ground', 7:'road', 8:'sidewalk', 9:'parking',
#           10:'rail track', 11:'building', 12:'wall', 13:'fence', 14:'guard rail',
#           15:'bridge', 16:'tunnel', 17:'pole', 18:'polegroup', 19:'traffic light',
#           20:'traffic sign', 21:'vegetation', 22:'terrain', 23:'sky', 24:'person',
#           25:'rider', 26:'car', 27:'truck', 28:'bus', 29:'caravan', 30:'trailer',
#           31:'train', 32:'motorcycle', 33:'bicycle', -1:'license plate'
# }


