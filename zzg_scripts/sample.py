'''
code by zzg - 2020-06-09

'''
##实现从a文件夹中随机采样n个图片并保存到b文件夹中

import random
import glob
import os
import cv2

src_img_dir = "a"
dst_img_dir = "b"

##读取图片路径
img_Lists = glob.glob(src_img_dir + '/*.jpg')
#print(img_Lists)

img_basenames = []
for item in img_Lists:
    img_basenames.append(os.path.basename(item))
    #print(img_basenames)

##得到所有原图片的名字
img_name = []
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_name.append(temp1)
print(img_name)

##随机采集的数量
img1_name = random.sample(img_name, 5)  
print(img1_name)

##保存到新的文件夹
for img in img1_name:
    img1 = cv2.imread(src_img_dir + '/' + img + '.jpg')
    cv2.imwrite(dst_img_dir + '/' + img + '.jpg',img1)

print("finished!!")


