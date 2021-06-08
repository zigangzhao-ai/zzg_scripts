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
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pdb

#the direction/path of Image,Label
src_img_dir = "image"
src_xml_dir = "xml"

img_Lists = glob.glob(src_img_dir + '/*.jpg')

img_basenames = []
for item in img_Lists:
    img_basenames.append(os.path.basename(item))
    
img_name = []
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_name.append(temp1)
 
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

for img in img_name:
    im = cv2.imread(src_img_dir + '/' + img + '.jpg')
    #read xml
    AnotPath = src_xml_dir + '/' + img + '.xml'
    tree = ET.ElementTree(file=AnotPath)  
    root = tree.getroot()
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
        ObjBndBoxSet.append(BndBoxLoc)    
    print("===========start rectangle bndbox==============")
    i = 0
    for x in ObjBndBoxSet:  
        [classname,x1,y1,x2,y2] = x 
        img1 = cv2.rectangle(im,(x1,y1),(x2,y2),(64,244,208),6)
        cv2.putText(img1, classname +'_' + '{}%'.format(i), (int(x1), int(y1+25)),
                            FONT, 1, (255, 255, 255), 4, cv2.LINE_AA)
        i += 4
    plt.imshow(img1)
    plt.show()
print("=======================finished!===================")
   