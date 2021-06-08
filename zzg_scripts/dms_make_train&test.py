
##按照9：1比例划分训练集和测试集，用于分类
import xml.etree.ElementTree as ET
import os
from os import getcwd
import shutil
import random
import numpy as np
dataset_dir = '/home/tanwensheng/work_space/tanwensheng/dataset/dms_withface_eye_mouth/strict/'
#dataset_dir = '/home/tanwensheng/work_space/tanwensheng/dataset/dms_withface_eye_mouth/expand/'




classes = ['call','fenxin','normal','smoke','tired']


def convert_annotation(xml_path, list_file):
    in_file = open(xml_path)
    tree=ET.parse(in_file)
    root = tree.getroot()
    empty = True
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        empty = False
    if(empty == True):
        list_file.write(" " + ",".join([str(a) for a in [0,0,0,0,classes.index('bg')]]))
wd = getcwd()

os.chdir(dataset_dir)
list_train_file = open('./dms_dataset_train.txt', 'w')
list_test_file = open('./dms_dataset_test.txt', 'w')
list_quant_file = open('./quant/dms_dataset_quant.txt', 'w')

for class_name in classes:
    dataset_path = os.path.join(dataset_dir,"all",class_name)
    if not os.path.exists(dataset_path):
        continue
    file_list=os.listdir(dataset_path)
    if not os.path.exists(os.path.join("./train/",class_name)):
	    os.mkdir(os.path.join("./train/",class_name)) 
    if not os.path.exists(os.path.join("./test/",class_name)):
	    os.mkdir(os.path.join("./test/",class_name)) 
    for filename in file_list:
    	#print(filename)
    	Allsuffix=os.path.splitext(filename)
    	file,suffix=Allsuffix
    	#print(suffix)
    	if suffix==str('.jpeg') or suffix==str('.jpg'):   
    	    img_path = os.path.join(dataset_path,filename)
    	    xml_path = os.path.join(dataset_path,file+'.xml') 
    	    #if not os.path.exists(xml_path):
    	    #    continue   	    
    	    if np.random.rand() < 0.9:
	            shutil.copy(img_path,os.path.join("./train/",class_name, filename))
	            shutil.copy(xml_path,os.path.join("./train/",class_name,file+'.xml'))
	            list_train_file.write(img_path)
	            convert_annotation(xml_path,list_train_file)
	            list_train_file.write('\n')

	            if np.random.rand() < 0.5:
	                shutil.copy(img_path,os.path.join("./quant/",filename))
	                list_quant_file.write(filename)
	                list_quant_file.write('\n')
    	    else:
	            #shutil.copy(xml_path,os.path.join("../mAP/xml/",file+'.xml'))
    	        #shutil.copy(img_path,os.path.join("../mAP/img/",filename))
	            shutil.copy(img_path,os.path.join("./test/",class_name, filename))
	            shutil.copy(xml_path,os.path.join("./test/",class_name, file+'.xml'))
	            list_test_file.write(img_path)
	            convert_annotation(xml_path,list_test_file)
	            list_test_file.write('\n')        
	            
list_train_file.close()
list_test_file.close()
list_quant_file.close()



