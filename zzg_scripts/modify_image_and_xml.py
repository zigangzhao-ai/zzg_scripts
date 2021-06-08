
'''
code by zzg 2020-05-30
'''
#!/usr/bin/python
# -*- coding: UTF-8 -*-
# get annotation object bndbox location

try:
    import xml.etree.cElementTree as ET  
except ImportError:
    import xml.etree.ElementTree as ET

import os,sys
import glob
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pdb

#the direction/path of Image,Label
src_img_dir = "/home/zigangzhao/zzg_project/batch_move/0618/test/e/smoke"
src_xml_dir = "/home/zigangzhao/zzg_project/batch_move/0618/test/e/smoke_xml"
dst_img_dir = "/home/zigangzhao/zzg_project/batch_move/0618/test/e/smoke_1"
dst_xml_dir = "/home/zigangzhao/zzg_project/batch_move/0618/test/e/smoke_xml_1"

if not os.path.exists(dst_img_dir):
    os.makedirs(dst_img_dir)

if not os.path.exists(dst_xml_dir):
    os.makedirs(dst_xml_dir)

img_Lists = glob.glob(src_img_dir + '/*.jpeg')
#print(img_Lists)
# rename = src_img_dir.split('/')[-1]
# print(rename)
rename = 'test'

img_basenames = []
for item in img_Lists:
    img_basenames.append(os.path.basename(item))
    #print(img_basenames)

img_name = []
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_name.append(temp1)
    print(img_name)

cnt = 0
for img in img_name:
    # im0 = Image.open((src_img_dir + '/' + img + '.jpeg'))
    # print(type(im0))
    # width, height = im0.size
    # print(im0.size)
    #print(width)
    ## read the scr_image
    im = cv2.imread(src_img_dir + '/' + img + '.jpeg')
    # print(type(im))
    print(im.shape[::-1])
    channels, width, height = im.shape[::-1]  ##get w and h
    # print(width, height)

    ##read the scr_xml
    AnotPath = src_xml_dir + '/' + img + '.xml'
    # print(AnotPath)
    tree = ET.ElementTree(file=AnotPath)  
    # print(tree)
    root = tree.getroot()
    # print(root)
    ObjectSet = root.findall('object')
    ObjBndBoxSet = []
    ObjBndBoxSet1 = {} 
    for Object in ObjectSet:
        ObjName = Object.find('name').text
        BndBox = Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text)#-1 
        y1 = int(BndBox.find('ymin').text)#-1
        x2 = int(BndBox.find('xmax').text)#-1
        y2 = int(BndBox.find('ymax').text)#-1
        BndBoxLoc = [ObjName,x1,y1,x2,y2]
        # print(x1,y1,x2,y2)
        ObjBndBoxSet.append(BndBoxLoc) 
        # if ObjName in ObjBndBoxSet:
        # 	ObjBndBoxSet1[ObjName].append(BndBoxLoc)#如果字典结构中含有这个类别了，那么这个目标框要追加到其值的末尾
        # else:
        # 	ObjBndBoxSet1[ObjName] = [BndBoxLoc]#如果字典结构中没有这个类别，那么这个目标框就直接赋值给其值吧   
        print(ObjBndBoxSet)
    
    #get the face 
    # [name,x01,y01,x02,y02] = ObjBndBoxSet[0]
    # print(len(ObjBndBoxSet))
    # img1 = cv2.rectangle(im,(x01,y01),(x02,y02),(255,0,0),2)
    # img2 = im[y01:y02, x01:x02]
   


    # plt.imshow(img2)
    # plt.show()

    # save the crop-image in dst_crop
    cv2.imwrite(dst_img_dir + '/' + rename + '_' + img + '.jpg',im) #rename + '_' +

    # rewrite xml to dst_xml
    xml = open((dst_xml_dir + '/' + rename + '_' + img + '.xml'), 'w')

    xml.write('<annotation>\n')
    xml.write('\t<folder>' + 'VOC2007' + '</folder>\n')
    xml.write('\t<filename>' + rename + '_' + str(img) + '.jpg' + '</filename>\n')
    xml.write('\t<source>\n')
    xml.write('\t\t<database>Unknown</database>\n')
    xml.write('\t</source>\n')
    xml.write('\t<size>\n')
    xml.write('\t\t<width>'+ str(width) + '</width>\n')
    xml.write('\t\t<height>'+ str(height) + '</height>\n')
    xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
    xml.write('\t</size>\n')
    xml.write('\t\t<segmented>0</segmented>\n')
      
    print("===========start rewrite bndbox==============")
    for x in ObjBndBoxSet:
        # print(x)
        [classname,x1,y1,x2,y2] = x   

        xml.write('\t<object>\n')
        xml.write('\t\t<name>'+ classname +'</name>\n')
        xml.write('\t\t<pose>Unspecified</pose>\n')
        xml.write('\t\t<truncated>1</truncated>\n')
        xml.write('\t\t<difficult>0</difficult>\n')
        xml.write('\t\t<bndbox>\n')
        xml.write('\t\t\t<xmin>' + str(x1) + '</xmin>\n')
        xml.write('\t\t\t<ymin>' + str(y1) + '</ymin>\n')
        xml.write('\t\t\t<xmax>' + str(x2) + '</xmax>\n')
        xml.write('\t\t\t<ymax>' + str(y2) + '</ymax>\n')
        xml.write('\t\t</bndbox>\n')
        xml.write('\t</object>\n')        
            
    xml.write('</annotation>')
    cnt += 1
    print(cnt)

print("=======================finished!===================")
