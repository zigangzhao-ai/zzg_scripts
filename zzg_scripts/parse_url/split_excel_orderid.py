#!/usr/bin
# Author       : zzg 
# Last modified: 2021-09-30 11:49
# Email        : 1415411655@qq.com

'''
cnt = 0
camera_id_all = []
for img in img_lists:
    cnt += 1
    print(cnt)
    bname1 = os.path.basename(img)
    tag1, _ = os.path.splitext(bname1) 
    tag2 = tag1.split('_')[-2]

'''
import os
import sys
import json
import numpy as np
import copy

import cv2
import pandas as pd
import urllib
import urllib.request as ur
import requests as req

decode_url = 'http://preview.i.utopia-smart-construction-site.home.ke.com/constructionsite/camera/get/preSigned/image/url?'

output_dir = "af_parse_img/1122"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

excel_file = '/root/zzg/parse_url/data_csv/0929_distinguish_liv_kitchen.xlsx'
df = pd.read_excel(excel_file, sheet_name='sheet1', header=0, usecols=[0,1,2,3,4,5])

data = df.values
row = data.shape[0]
col = data.shape[1]
print(row, col)
# print(data[0, :])
cnt = 0
for i in range(0, row, 50):
    
    order_id = data[i,1]
    room_type = data[i,2]
    device_id = data[i,3]

    order_id_file = output_dir + '/' +  str(order_id)
    if not os.path.exists(order_id_file):
        os.mkdir(order_id_file)

    device_id_file = order_id_file + '/' + device_id
    if not os.path.exists(device_id_file):
        os.mkdir(device_id_file)

    img_url_bef = data[i,4]
    bname = os.path.basename(img_url_bef)
    tag, _ = os.path.splitext(bname) 

    img_create_time = (data[i,5]).replace(' ', '-').replace(':', '-')
  
    url_params = {'originUrl': img_url_bef}       
    resp = req.get(decode_url, params=url_params)
    cons_res = json.loads(resp.content)
    
    if cons_res['code'] == 2000:
        cnt += 1
        print(cnt)
        img_url = cons_res['data']
        response = req.get(img_url)
        img_src = cv2.imdecode(np.frombuffer(response.content, np.uint8), flags=cv2.IMREAD_COLOR)

        cv2.imwrite(os.path.join(device_id_file, str(device_id)+'_'+ room_type +'_'+ tag+'.jpg'), img_src)

