'''
code by zzg-2020-06-02
'''

import os
import numpy as np
import codecs
import pandas as pd
import json
from glob import glob
import cv2
import shutil
from sklearn.model_selection import train_test_split
from IPython import embed
#1.标签路径
csv_file = "train.csv"
saved_path = "VOC2007/"    #保存路径
image_save_path = "./JPEGImages/"
image_raw_parh = "/home/zigangzhao/zzg_project/PyTorch-YOLOv3/dataloader/train/"
#2.创建要求文件夹
if not os.path.exists(saved_path + "Annotations"):
    os.makedirs(saved_path + "Annotations")
if not os.path.exists(saved_path + "JPEGImages/"):
    os.makedirs(saved_path + "JPEGImages/")
if not os.path.exists(saved_path + "ImageSets/Main/"):
    os.makedirs(saved_path + "ImageSets/Main/")
    
#3.获取待处理文件
data = pd.read_csv(csv_file,header=None,index_col=False,
                  names=['image_id','width','height','bbox','source'])

##合并相同名字的行
data_lite = data[['image_id','bbox']]
# print(data_lite)
data_lite['bbox'] = data_lite['bbox'].apply(lambda x: ','+ x)
data1 = data_lite.groupby(by='image_id').sum()
# data1 = data_lite.groupby(by='image_id')['bbox'].sum()
data1['bbox'] = data1['bbox'].apply(lambda x : x[1:])
data1 = data1[0:3373]  ##去除最后一行标签
# print(data1)

total_csv_annotations = {}
for row in data1.itertuples():
    # print(row[0],row[1])
    total_csv_annotations[row[0]] = row[1]

##适用于没用相同名字的csv
# total_csv_annotations = {}
# annotations = pd.read_csv(csv_file,header=None).values
# print(annotations )

#     key = annotation[0].split(os.sep)[-1]
#     value = np.array(annotation[3:])
#     value = value[0]
#     # print(key)
#     # print(type(value))
#     # print(value)
#     # print(total_csv_annotations.keys())
    
#     # total_csv_annotations[key] = value   
#     total = total_csv_annotations
# print(total)

#4.读取标注信息并写入 xml
# print(total_csv_annotations.items())

count = 0
for filename,label in total_csv_annotations.items():
    #embed()
    # print(filename)
    count += 1
    print(count)
    height, width, channels = cv2.imread(image_raw_parh + filename + '.jpg').shape
    #embed()
    with codecs.open(saved_path + "Annotations/"+filename+'.xml',"w","utf-8") as xml:
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'VOC2007' + '</folder>\n')
        xml.write('\t<filename>' + filename + '.jpg' + '</filename>\n')
        xml.write('\t<source>\n')
        xml.write('\t\t<database>Unknown</database>\n')
        xml.write('\t</source>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>'+ str(width) + '</width>\n')
        xml.write('\t\t<height>'+ str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        if isinstance(label,float):
            ## 空白
            xml.write('</annotation>')
            continue
        # print(label)
        label = label.replace('[','').replace(']','').replace(' ', '').split(',')
        # print(label)
        box_cnt = len(label) // 4
        # print(label[3])
        for i in range(box_cnt):

            xmin = int(float(label[i*4]))
            ymin = int(float(label[i*4+1]))
            width = int(float(label[i*4+2]))
            height= int(float(label[i*4+3]))
            xmax = xmin + width
            ymax = ymin + height
            # classname = 'wheat'
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                xml.write('\t<object>\n')
                xml.write('\t\t<name>'+'wheat'+'</name>\n')
                xml.write('\t\t<pose>Unspecified</pose>\n')
                xml.write('\t\t<truncated>1</truncated>\n')
                xml.write('\t\t<difficult>0</difficult>\n')
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
                xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
                xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
                xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')
                print(filename,xmin,ymin,xmax,ymax)
        xml.write('</annotation>')
        
#6.split files for txt
txtsavepath = saved_path + "ImageSets/Main/"
ftrainval = open(txtsavepath+'/trainval.txt', 'w')
# ftest = open(txtsavepath+'/test.txt', 'w')
ftrain = open(txtsavepath+'/train.txt', 'w')
fval = open(txtsavepath+'/val.txt', 'w')
total_files = glob(saved_path+"./Annotations/*.xml")
total_files = [i.split("/")[-1].split(".xml")[0] for i in total_files]
#test_filepath = ""
for file in total_files:
    ftrainval.write(file + "\n")

# move images to voc JPEGImages folder
for image in glob(image_raw_parh+"/*.jpg"):
    shutil.copy(image,saved_path+image_save_path)

train_files,val_files = train_test_split(total_files,test_size=0.15,random_state=42)

for file in train_files:
    ftrain.write(file + "\n")
#val
for file in val_files:
    fval.write(file + "\n")

ftrainval.close()
ftrain.close()
fval.close()
