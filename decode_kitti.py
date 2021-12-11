'''
Created on Sep 9, 2017

@author: jyarde
'''
import cv2
import numpy as np

KITTI_XLATE = {
    0: 1,
    1: 101
}

KITTI_PALETTE = {
    0: (255, 0, 0),
    1: (255, 0, 255)
}

KITTI_LABELS = {"background": 0, "road": 1}

TARGET_CLASS_NAMES = ["background", "road"]

kitti_target_classes = len(TARGET_CLASS_NAMES)

def fix_transform_fills(label, cval=1):
    '''
    Areas filled with 1's because of geometric transforms need
    to be filled with background color
    label - transformed mask image
    cval  - fill value used for transforms
    '''
    m = np.copy(label)
    channels = []
    for i in range(3):
        c = m[:,:,i]
        c[c == cval] = KITTI_PALETTE[0][i]
        channels.append(c)
    return np.stack(channels, -1)

def decode(label):
    code = np.zeros(label.shape)
    label = label/255
    code[:,:,0] = label[:,:,0] + label[:,:,1]*10 + label[:,:,2]*100
    decoded = np.zeros(label.shape[:2]+(kitti_target_classes,) )
    for i in range(kitti_target_classes):
        layer = np.zeros(label.shape[:2])
        # Needed because Keras ImageGenerator reorders classes alphanumerically
        classid = KITTI_LABELS[TARGET_CLASS_NAMES[i]]
        layer[code[:,:,0]==KITTI_XLATE[classid]] = 1
        decoded[:,:,i] = layer
    return decoded

def decode_class(label, classid=None):
    code = np.zeros(label.shape)
    label = label/64
    code[:,:,0] = label[:,:,0] + label[:,:,1]*10 + label[:,:,2]*100
    layer = np.zeros(label.shape[:2])
    target_class = TARGET_CLASS_NAMES[classid]
    kitti_classid = KITTI_LABELS[target_class]
    layer[code[:,:,0]==KITTI_XLATE[kitti_classid]] = 1
    return layer
