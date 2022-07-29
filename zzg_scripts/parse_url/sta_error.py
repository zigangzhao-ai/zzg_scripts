#!/usr/bin
# Author       : zzg 
# Last modified: 2021-10-26 14:41
# Email        : 1415411655@qq.com

import os
import sys
import json
import numpy as np
import copy

import cv2
import pandas as pd
import urllib
import urllib.request as ur

from matplotlib import pyplot as plt
import matplotlib


#绘制箱形图
def drawBox(heights):
    #创建箱形图
    #第一个参数为待绘制的定量数据
    #第二个参数为数据的文字说明
    plt.boxplot([heights], labels=['Heights'])
    plt.title('Heights Of Male Students')
    plt.show()

#绘制直方图
def drawHist(data):
    """
    绘制直方图
    data:必选参数，绘图数据
    bins:直方图的长条形数目，可选项，默认为10
    facecolor:长条形的颜色
    edgecolor:长条形边框的颜色
    alpha:透明度
    """
    width = 15
    bins = [i for i in range(int(min(data)), int(max(data)) + width, width)]
  
    plt.hist(data, bins=bins, density=0, facecolor="blue", edgecolor="black", alpha=0.7, stacked=True)
    plt.xticks(ticks=bins)

    # 显示横轴标签
    plt.xlabel("interval")
    # 显示纵轴标签
    plt.ylabel("frequency")
    # 显示图标题
    plt.title("frequency_hist")
    
    plt.savefig("hist.jpg")
    plt.grid()
    plt.show()


decode_url = 'http://preview.i.utopia-smart-construction-site.home.ke.com/constructionsite/camera/get/preSigned/image/url?'

# save_dir = "af_parse_img/281878"
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)

excel_file = '/Users/zzg_beike/Desktop/test_code/parse_url/data_csv/1026_sta.xlsx'
df = pd.read_excel(excel_file, sheet_name='sheet1', header=0, usecols=[0,1,2,3,4,5])

data = df.values
row = data.shape[0]
col = data.shape[1]
print(row, col)
# print(data[0, :])
ext_error_all = []
ext_error_part = []
cnt = 0
for i in range(0, row, 1):
    if isinstance(data[i, 4], str):
        cnt += 1
        print(cnt)
        extrinsic_param = eval(data[i, 4])
        error = extrinsic_param['extrinsic_error']
        print(error)
        ext_error_all.append(error/4)

        if error < 1000:
            ext_error_part.append(error/4)


min_value, max_value = min(ext_error_all), max(ext_error_all)

print(min_value, max_value)
print(len(ext_error_all), len(ext_error_part))
# drawBox(ext_error_all)
# print(sorted(ext_error_all)[:-2])
ext_error_part = np.array(ext_error_part)
drawHist(ext_error_part)
