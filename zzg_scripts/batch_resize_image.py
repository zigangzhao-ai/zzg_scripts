#coding=utf-8
'''
code by zzg 2020-04-07
'''
#提取目录下所有图片,更改尺寸后保存到另一目录
from PIL import Image
import os.path
import glob

def convertjpg(jpgfile,outdir,width=128,height=128):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)   
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)
for jpgfile in glob.glob("/home/zigangzhao/zzg_project/CNN-mask-classification/data/train/*.jpg"):
    convertjpg(jpgfile,"/home/zigangzhao/zzg_project/CNN-mask-classification/data/train02/")

