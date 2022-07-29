#!/usr/bin
# Author       : zzg 
# Last modified: 2021-09-28 10:36
# Email        : 1415411655@qq.com

import os
import glob
import pandas as pd
import shutil

scr_img = "/root/zzg/parse_url/struct_test/structure_test_0927"
output_dir = "/root/zzg/parse_url/struct_test/struct_test_class"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

img_lists = glob.glob(scr_img + '/*.jpg')
excel_file = '/root/zzg/parse_url/data_csv/11976875.xlsx'
df = pd.read_excel(excel_file, sheet_name='sheet1', header=0, usecols=[0,1,2,3,4,5])

data = df.values[150000:]
row = data.shape[0]
col = data.shape[1]
print(row, col)

cnt = 0
camera_id_all = []
for img in img_lists:
    cnt += 1
    print(cnt)
    bname1 = os.path.basename(img)
    tag1, _ = os.path.splitext(bname1) 
    tag2 = tag1.split('_')[-2]

    for i in range(0, row, 20):
        device_id = data[i,3]
        camera_id_all.append(device_id)
           
        camera_id_file = output_dir + '/' + device_id
        if not os.path.exists(camera_id_file):
            os.mkdir(camera_id_file)

        img_create_time = (data[i,5]).replace(' ', '-').replace(':', '-')
        img_url_bef = data[i,4]

        if tag2 in img_url_bef:
            #print('++++')
            new_path = camera_id_file + '/' + str(device_id) + '_' + str(img_create_time)+ '_'+ tag2 + '.jpg' 
            shutil.copy(img, new_path)





  