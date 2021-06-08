import skimage.io as io
import os
#import cv2
 
data_dir='/workspace1/zigangzhao/JPEGImages'
str=data_dir + '/*.JPEG'
coll = io.ImageCollection(str)
#print(coll[1])
#print(len(coll))
#io.imshow(coll[1])
#io.show()
file_list= []
for file in os.listdir(data_dir): 
    if file.endswith(".JPEG"): 
        write_name = file
        filename = os.path.splitext(write_name)[0]
        #print(filename)
        file_list.append(filename) 
        sorted(file_list) 
#print(file_list)
 
result=[]
with open('/workspace1/zigangzhao/B/trainval.txt','r') as f:
    for line in f:
        line=line.strip('\n')
        result.append(line)
#print(result)
 
for i in range(0,900):
    #print(file_list[i])
    for j in range(0,810):
        #print("i=",i)
        #print("j=",j)
        #print(file_list[i])
        #print(result[j])
        #a=[]
        #print(j)
        if file_list[i]==result[j]:
            print(file_list[j])
            io.imsave("/workspace1/zigangzhao/B/train/"+ result[j] + ".JPEG",coll[i])
