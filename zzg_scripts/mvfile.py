# coding:utf-8
'''
code by zzg 2020.04.02
'''
#移动一个文件夹下的所有文件夹到另一个文件夹

import os
import shutil
 
src_dir = 'video'
dst_dir = 'zzgvideo'


#创建文件夹
if not os.path.exists(src_dir):
    os.makedirs(src_dir)

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

if os.path.exists(src_dir):
    for root, dirs, files in os.walk(src_dir):
        
        # print("root:",root)#文件夹路径
        print("dirs:",dirs)#文件夹名称
        # print("files:",files)#文件名
     
        for file in files[937:]:#遍历每一个文件
            print(file)
      
        for dir in dirs:
            dir_file = os.path.join(src_dir, dir)
        shutil.move(dir_file, dst_dir)#移动文件
