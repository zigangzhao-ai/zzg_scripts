
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
src_img_dir = "image"
src_xml_dir = "xml"
dst_img_dir = "image-crop"
dst_xml_dir = "xml-crop"

img_Lists = glob.glob(src_img_dir + '/*.jpeg')
#print(img_Lists)

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
    print(type(im))
    print(im.shape)
    width, height = im.shape[:2][::-1]  ##get w and h
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
    [name,x01,y01,x02,y02] = ObjBndBoxSet[0]
    # print(len(ObjBndBoxSet))
    # img1 = cv2.rectangle(im,(x01,y01),(x02,y02),(255,0,0),2)
    img2 = im[y01:y02, x01:x02]
    out = img.resize((224,224),Image.ANTIALIAS)
    out1 = img2.resize((224,224),Image.ANTIALIAS)


    # plt.imshow(img2)
    # plt.show()

    # save the crop-image in dst_crop
    cv2.imwrite(dst_img_dir + '/' + img + '.jpeg',img2)

    # rewrite xml to dst_xml
    xml_file = open((dst_xml_dir + '/' + img + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('<folder>VOC2007</folder>\n')
    xml_file.write('<filename>' +str(img) + '.jpeg' + '</filename>\n')
    xml_file.write('<size>\n')
    xml_file.write('<width>' + str(width) + '</width>\n')
    xml_file.write('<height>' + str(height) + '</height>\n')
    xml_file.write('<depth>3</depth>\n')
    xml_file.write('</size>\n')
      
    print("===========start rewrite bndbox==============")
    for x in ObjBndBoxSet[1:]:
        print(x)
        [classname,x1,y1,x2,y2] = x 
        x1 = x1 - x01
        y1 = y1 - y01
        x2 = x2 - x01
        y2 = y2 - y01    
        xml_file.write('<object>\n')
        xml_file.write('<name>' + classname + '</name>\n')
        xml_file.write('<pose>Unspecified</pose>\n')
        xml_file.write('<truncated>0</truncated>\n')
        xml_file.write('<difficult>0</difficult>\n')
        xml_file.write('<bndbox>\n')
        xml_file.write('<x1>' + str(x1) + '</x1>\n')
        xml_file.write('<y1>' + str(y1) + '</y1>\n') 
        xml_file.write('<x2>' + str(x2) + '</x2>\n')
        xml_file.write('<y2>' + str(y2) + '</y2>\n')
        xml_file.write('</bndbox>\n')
        xml_file.write('</object>\n')         
            
    xml_file.write('</annotation>')
    cnt + = 1
    print(cnt)

print("=======================finished!===================")
   