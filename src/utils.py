# coding: utf-8
__author__ = 'RocketFlash: https://github.com/RocketFlash'

import cv2
from matplotlib import pyplot as plt

def show_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    plt.title('image')
    plt.show()

def draw_bboxes(image, bboxes, color=(255,0,0)):
    for bbox in bboxes:
        cv2.rectangle(image,(bbox[0], bbox[1]),(bbox[2], bbox[3]),color,2)
