# -*- coding: utf-8 -*-
"""
Created on 04.13 2020
@author: zzg
"""
#读取一张图片随机剪裁

import random
import cv2
import numpy as np

from matplotlib import pyplot as plt

path1 = 'output/'
def random_crop(image, min_ratio=0.6, max_ratio=1.0):

    h, w = image.shape[:2]
    
    ratio = random.random()
    
    scale = min_ratio + ratio * (max_ratio - min_ratio)
    
    new_h = int(h*scale)    
    new_w = int(w*scale)
    
    y = np.random.randint(0, h - new_h)    
    x = np.random.randint(0, w - new_w)
    
    image = image[y:y+new_h, x:x+new_w, :]
    
    return image


img = cv2.imread("13026.jpg")
print(img.shape)
cv2.imwrite("./output1/initial.jpg", img)
img1 = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)
print(img1.shape)
cv2.imwrite("./output1/resize.jpg", img1)

i = 0
for i in range(0,20):
    img2 = random_crop(img1)
    print(img2.shape)
    # plt.imshow(img2)
    cv2.imwrite("./output1/photo_{}.jpg".format(i), img2)
    i += 1
    # plt.show()

print("finished!!")
