import os,sys
import glob
from PIL import Image

# the direction/path of Image,Label
src_img_dir = "/workspace1/zigangzhao/tiny/bird"
src_txt_dir = "/workspace1/zigangzhao/tiny"
src_xml_dir = "/workspace1/zigangzhao/tiny"

img_Lists = glob.glob(src_img_dir + '/*.JPEG')
#print(img_Lists)

img_basenames = []
for item in img_Lists:
    img_basenames.append(os.path.basename(item))
    #print(img_basenames)

img_name = []
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_name.append(temp1)
    #print(img_name)

for img in img_name:
    im = Image.open((src_img_dir + '/' + img + '.JPEG'))
    width, height = im.size
    #print(width)
    
    
    #gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()
    gt = open(src_txt_dir + '/' + 'bird_gt.txt').read().splitlines()
    #print(gt)
    #print(len(gt))
    #img = int(img) + 180
    #img = "%06d" % img
    
    print(img)
    xml_file = open((src_xml_dir + '/' + img + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('<folder>VOC2007</folder>\n')
    xml_file.write('<filename>' + str(img) + '.JPEG' + '</filename>\n')
    xml_file.write('<size>\n')
    xml_file.write('<width>' + str(width) + '</width>\n')
    xml_file.write('<height>' + str(height) + '</height>\n')
    xml_file.write('<depth>3</depth>\n')
    xml_file.write('</size>\n')
    fg = img
    print(fg)
    for index in range(len(gt)):
        if index == int(fg)-1 :
            spt = gt[index].split(' ')
            xml_file.write('<object>\n')
            #xml_file.write('<name>' + str(spt[0]) + '</name>\n')
            xml_file.write('<name>' + 'bird' + '</name>\n')
            xml_file.write('<pose>Unspecified</pose>\n')
            xml_file.write('<truncated>0</truncated>\n')
            xml_file.write('<difficult>0</difficult>\n')
            xml_file.write('<bndbox>\n')
            xml_file.write('<xmin>' + str(spt[1]) + '</xmin>\n')
            xml_file.write('<ymin>' + str(spt[2]) + '</ymin>\n')
            xml_file.write('<xmax>' + str(spt[3]) + '</xmax>\n')
            xml_file.write('<ymax>' + str(spt[4]) + '</ymax>\n')
            xml_file.write('</bndbox>\n')
            xml_file.write('</object>\n')

    xml_file.write('</annotation>')
