
import os
import random  
  
trainval_percent = 0.8  #trainval占比例多少    #0.8
train_percent = 0.7  #test数据集占比例多少
xmlfilepath = '/workspace1/test/faster-rcnn.pytorch/data/VOCdevkit/VOC2007/Annotations'  
txtsavepath = '/workspace1/test/faster-rcnn.pytorch/data/VOCdevkit/VOC2007/ImageSets/Main'  
total_xml = os.listdir(xmlfilepath)  
  
num=len(total_xml)  
list=range(num)  
tv=int(num*trainval_percent)  
tr=int(tv*train_percent)  
trainval= random.sample(list,tv)  
train=random.sample(trainval,tr)  
  
ftrainval = open('/workspace1/test/faster-rcnn.pytorch/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', 'w')  
ftest = open('/workspace1/test/faster-rcnn.pytorch/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'w')  
ftrain = open('/workspace1/test/faster-rcnn.pytorch/data/VOCdevkit/VOC2007/ImageSets/Main/train.txt', 'w')  
fval = open('/workspace1/test/faster-rcnn.pytorch/data/VOCdevkit/VOC2007/ImageSets/Main/val.txt', 'w')  
  
for i  in list:  
    name=total_xml[i][:-4]+'\n'  
    if i in trainval:  
        ftrainval.write(name)  
        if i in train:  
            ftrain.write(name)  
        else:  
            fval.write(name)  
    else:  
        ftest.write(name)  
  
ftrainval.close()  
ftrain.close()  
fval.close()  
ftest .close()
