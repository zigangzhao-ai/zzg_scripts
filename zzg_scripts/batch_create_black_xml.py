# coding:utf-8
'''
code by zzg 2020-04-04

'''
##针对负样本的图片批量生成空白标签

import os,sys
import glob
from PIL import Image
import pdb

# the direction/path of Image,Label
src_img_dir = "images/"
src_xml_dir = "xml/"

img_Lists = glob.glob(src_img_dir + '/*.jpg')
# print(img_Lists)

img_basenames = []
for item in img_Lists:
    img_basenames.append(os.path.basename(item))
# print(img_basenames)

img_name = []
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_name.append(temp1)
# print(img_name)


#pdb.set_trace()
for img in img_name:
    im = Image.open((src_img_dir + '/' + img + '.jpg'))
    width, height = im.size
    #print(width)
    
    xml_file = open((src_xml_dir + '/' + img + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('<folder>VOC2007</folder>\n')
    xml_file.write('<filename>' +str(img) + '.jpg' + '</filename>\n')
    xml_file.write('<source>\n')
    xml_file.write('<database>' + 'Unknown' + '</database>\n')
    xml_file.write('</source>\n')
    xml_file.write('<size>\n')
    xml_file.write('<width>' + str(width) + '</width>\n')
    xml_file.write('<height>' + str(height) + '</height>\n')
    xml_file.write('<depth>3</depth>\n')
    xml_file.write('</size>\n')
    #print(len(gt))
    xml_file.write('</annotation>')

print("finshed convert!!")

    
    