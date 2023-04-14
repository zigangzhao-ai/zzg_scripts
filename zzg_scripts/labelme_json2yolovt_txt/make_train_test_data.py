#!/usr/bin
# Author       : zzg 
# Last modified: 2023-03-09 15:29

"""
按照yolov5训练要求，按照8:2划分训练集和测试集
"""

import os
import random

time_flag = "0403"

out_dir10 = "{}_train/train_{}/train/images".format(time_flag, time_flag)
out_dir11= "{}_train/train_{}/train/labels".format(time_flag, time_flag)
out_dir20 = "{}_train/train_{}/val/images".format(time_flag, time_flag)
out_dir21 = "{}_train/train_{}/val/labels".format(time_flag, time_flag)

os.makedirs(out_dir10, exist_ok=True)
os.makedirs(out_dir11, exist_ok=True)
os.makedirs(out_dir20, exist_ok=True)
os.makedirs(out_dir21, exist_ok=True)

src_dir = "{}_train/txt".format(time_flag)
img_dir = "0403_all"
txt_names = os.listdir(src_dir)

l = len(txt_names)
sel_names = random.sample(txt_names, int(l*0.8)) #0.90
print(sel_names)
for txt_name in txt_names:
    txt_pth = src_dir + '/' + txt_name
    img_pth = img_dir + '/' + txt_name.replace('txt', 'jpg')
    if txt_name in sel_names:  
        print("---", txt_name)
        cmd1 = 'cp -r {} {}'.format(img_pth, out_dir10)
        cmd2 = 'cp -r {} {}'.format(txt_pth, out_dir11)

    else:
        cmd1 = 'cp -r {} {}'.format(img_pth, out_dir20)
        cmd2 = 'cp -r {} {}'.format(txt_pth, out_dir21)
    os.system(cmd1)
    os.system(cmd2)
# print('----', i)