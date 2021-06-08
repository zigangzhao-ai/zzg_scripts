#coding=utf-8
'''
code by zzg   2020-04-07
'''

import os
import re
import sys
path = r"/home/zigangzhao/zzg_project/CNN-mask-classification/data/原始图片/mask/mask_dataset_hamlin/have_mask/"
fileList = os.listdir(path)  # 待修改文件夹
print("修改前：" + str(fileList))  # 输出文件夹中包含的文件
os.chdir(path)  # 将当前工作目录修改为待修改文件夹的位置
num = 0  # 名称变量
for fileName in fileList:  # 遍历文件夹中所有文件
    pat = ".+\.(jpg|jpeg|JPG)"  # 匹配文件名正则表达式
    pattern = re.findall(pat, fileName)  # 进行匹配
    # print(pattern)
    print('num：', num, 'filename:', fileName)
    os.rename(fileName, ('1' + '_' + str(num) + '.'+ pattern[0]))  # 文件重新命名
    num = num + 1  # 改变编号，继续下一项
print("---------------------------------------------------")
sys.stdin.flush()  # 刷新
print("修改后：" + str(os.listdir(path)))  # 输出修改后文件夹中包含的文件