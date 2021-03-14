# coding: utf-8
__author__ = 'RocketFlash: https://github.com/RocketFlash'

import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.title('image')
    plt.show()

def draw_bboxes(image, bboxes, mask=None, color=(255,0,0)):

    result_image = image.copy()
    result_mask = np.zeros_like(result_image)

    if mask is not None:
        ids = np.unique(mask)
        ids = ids[ids != 0]
        for id_i in ids:
            id_color = list(np.random.choice(range(256), size=3))
            result_mask[mask==id_i] = id_color
        cv2.addWeighted(result_mask, 0.4, result_image, 1, 0, result_image)


    for bbox in bboxes:
        if len(bbox)==5:
            cv2.rectangle(result_image,(bbox[1], bbox[2]),(bbox[3], bbox[4]),color,2)
        else:
            cv2.rectangle(result_image,(bbox[0], bbox[1]),(bbox[2], bbox[3]),color,2)
    
    if mask is None:
        return result_image, None
    else:
        return result_image, result_mask
