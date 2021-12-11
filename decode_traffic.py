'''
Created on Sep 9, 2017

@author: jyarde
'''
import cv2
import numpy as np

TRAFFIC_XLATE = {
    0: 0,
    1: 2,
    2: 20,
    3: 22,
    4: 200
}

TRAFFIC_PALETTE = {
    0: (0, 0, 0),
    1: (128, 0, 0),
    2: (0, 128, 0),
    3: (128, 128, 0),
    4: (0, 0, 128)
}

TRAFFIC_LABELS = { "background": 0, "red": 1, "yellow": 2, "green": 3, "off":4 }

TARGET_CLASS_NAMES = ["background", 
                      "red", 
                      "yellow", 
                      "green", 
                      "off"]

traffic_target_classes = len(TARGET_CLASS_NAMES) - 1

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
        c[c == cval] = TRAFFIC_PALETTE[0][i]
        channels.append(c)
    return np.stack(channels, -1)


def decode(label):
    code = np.zeros(label.shape)
    label = label.astype("uint8")/64
    code[:,:,0] = label[:,:,0] + label[:,:,1]*10 + label[:,:,2]*100
    decoded = np.zeros(label.shape[:2]+(traffic_target_classes,))
    for i in range(traffic_target_classes):
        layer = np.zeros(label.shape[:2])
        # Needed because Keras ImageGenerator reorders classes alphanumerically
        voc_classid = TRAFFIC_LABELS[TARGET_CLASS_NAMES[i]]
        layer[code[:,:,0]==TRAFFIC_XLATE[voc_classid]] = 1
        decoded[:,:,i] = layer
    return decoded

def decode_class(label, classid=None):
    code = np.zeros(label.shape)
    label = label/64
    code[:,:,0] = label[:,:,0] + label[:,:,1]*10 + label[:,:,2]*100
    layer = np.zeros(label.shape[:2])
    target_class = TARGET_CLASS_NAMES[classid]
    voc_classid = TRAFFIC_LABELS[target_class]
    layer[code[:,:,0]==TRAFFIC_XLATE[voc_classid]] = 1
    return layer
