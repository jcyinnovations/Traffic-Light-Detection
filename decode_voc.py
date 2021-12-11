'''
Created on Sep 9, 2017

@author: jyarde
'''
import cv2
import numpy as np

PASCAL_XLATE = {
    0: 0,
    1: 2,
    2: 20,
    3: 22,
    4: 200,
    5: 202,
    6: 220,
    7: 222,
    8: 1,
    9: 3,
    10: 21,
    11: 23,
    12: 201,
    13: 203,
    14: 221,
    15: 223,
    16: 10,
    17: 12,
    18: 30,
    19: 32,
    20: 210
}

PASCAL_PALETTE = {
    0: (0, 0, 0),
    1: (128, 0, 0),
    2: (0, 128, 0),
    3: (128, 128, 0),
    4: (0, 0, 128),
    5: (128, 0, 128),
    6: (0, 128, 128),
    7: (128, 128, 128),
    8: (64, 0, 0),
    9: (192, 0, 0),
    10: (64, 128, 0),
    11: (192, 128, 0),
    12: (64, 0, 128),
    13: (192, 0, 128),
    14: (64, 128, 128),
    15: (192, 128, 128),
    16: (0, 64, 0),
    17: (128, 64, 0),
    18: (0, 192, 0),
    19: (128, 192, 0),
    20: (0, 64, 128),
}

VOC_LABELS = {"background": 0,
                "aeroplane": 1,
                "bicycle": 2,
                "bird": 3,
                "boat": 4,
                "bottle": 5,
                "bus": 6,
                "car": 7,
                "cat": 8,
                "chair": 9,
                "cow": 10,
                "diningtable": 11,
                "dog": 12,
                "horse": 13,
                "motorbike": 14,
                "person":  15,
                "pottedplant": 16,
                "sheep": 17,
                "sofa": 18,
                "train": 19,
                "tvmonitor": 20
                }

TARGET_CLASS_NAMES = ["background", 
                      "aeroplane", 
                      "horse", 
                      "motorbike", 
                      "person", 
                      "sheep", 
                      "train",
                      "bicycle", 
                      "bird", 
                      "boat", 
                      "bus", 
                      "car", 
                      "cat", 
                      "cow", 
                      "dog"]

def decode(label):
    code = np.zeros(label.shape)
    label = label/64
    code[:,:,0] = label[:,:,0] + label[:,:,1]*10 + label[:,:,2]*100
    #code = code.astype("uint8")
    decoded = np.zeros(label.shape[:2]+(15,))
    for i in range(15):
        layer = np.zeros(label.shape[:2])
        # Needed because Keras ImageGenerator reorders classes alphanumerically
        voc_classid = VOC_LABELS[TARGET_CLASS_NAMES[i]]
        layer[code[:,:,0]==PASCAL_XLATE[voc_classid]] = 1
        decoded[:,:,i] = layer
    return decoded

def decode_class(label, classid=None):
    code = np.zeros(label.shape)
    label = label/64
    code[:,:,0] = label[:,:,0] + label[:,:,1]*10 + label[:,:,2]*100
    #print("Unique values", np.unique(code  a))
    layer = np.zeros(label.shape[:2])
    target_class = TARGET_CLASS_NAMES[classid]
    voc_classid = VOC_LABELS[target_class]
    #print("Target Class", classid, target_class, voc_classid, PASCAL_XLATE[voc_classid], label.shape)
    layer[code[:,:,0]==PASCAL_XLATE[voc_classid]] = 1
    return layer
