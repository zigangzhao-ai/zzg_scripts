#!/usr/bin
# Author       : zzg 
# Last modified: 2021-11-22 12:06
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
''
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

output_dir = "af_parse_img/img_1125"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

excel_file = '/Users/zzg_beike/Desktop/test_code/parse_url/data_csv/1125_debug.xlsx'
#0-bim_json_url  1-create_time  2-cur_img_url  3-device_id  4-error_code  5-error_msg  8-order_id 9-room_id 
df = pd.read_excel(excel_file, sheet_name='result', header=0, usecols=[0,1,2,3,4,5,6,7,8,9,10,11]) 

data = df.values

row = data.shape[0]
col = data.shape[1]
print(row, col)
# print(data[0, :])
cnt = 0
error_502 = 0
error_503 = 0
error_0_0 = 0
error_0_1 = 0


for i in range(0, row):
    
    bim3d_json_url = data[i,7]
    # create_time = data[i, 1]
    cur_img_url = data[i,10]
    device_id = data[i,4]
    error_code = data[i,2]
    error_msg = data[i,3]
    flag = data[i,1]
    order_id = data[i,5]
    pre_img_url = data[i,9]
    room_id = data[i,6]

    if error_code == 502:
        error_502 += 1
    elif error_code == 503:
        error_503 += 1
    elif error_code == 0:
        if flag == 0:
            error_0_0 += 1
        elif flag == 1:
            error_0_1 += 1

    error_msg_file = output_dir + '/' + str(error_code)+ '_' + str(error_msg)
    if not os.path.exists(error_msg_file):
        os.mkdir(error_msg_file)

    order_id_file = error_msg_file + '/' + str(order_id)
    if not os.path.exists(order_id_file):
        os.mkdir(order_id_file)

    bname = os.path.basename(cur_img_url)
    tag, _ = os.path.splitext(bname) 

    url_params = {'originUrl': cur_img_url}       
    resp = req.get(decode_url, params=url_params)
    cons_res = json.loads(resp.content)

    
    pre_params = {'originUrl': pre_img_url}
    pre_resp = req.get(decode_url, params=pre_params)
    pre_cons_res = json.loads(pre_resp.content)
 
    if cons_res['code']==2000 and pre_cons_res['code']==2000:
        cnt += 1
        print(cnt)
        img_url = cons_res['data']
        response = req.get(img_url) 
        img_src = cv2.imdecode(np.frombuffer(response.content, np.uint8), flags=cv2.IMREAD_COLOR)
        
        cv2.imwrite(os.path.join(order_id_file, str(device_id)+'_'+ str(room_id) +'_'+ tag+'_cur'+'.png'), img_src)

    
        pre_img_url = pre_cons_res['data']
        if len(pre_img_url) > 10:
            pre_response = req.get(pre_img_url) 
            pre_img = cv2.imdecode(np.frombuffer(pre_response.content, np.uint8), flags=cv2.IMREAD_COLOR)
        
            cv2.imwrite(os.path.join(order_id_file, str(device_id)+'_'+ str(room_id) +'_'+ tag+'_pre'+'.png'), pre_img)
            
        response_bim3d = req.get(bim3d_json_url)

    with open(os.path.join(order_id_file, 'bim3d.json'), 'wb') as f:
        f.write(response_bim3d.content)
        f.close()  
        # an order_id <--> a bim3d_json

print(error_502, error_503, error_0_0, error_0_1)
   


        # with open(os.path.join(order_id_file, str(device_id)+'_'+ str(room_id) +'_'+ tag+'.json'), 'wb') as f:
        #     f.write(response_bim3d.content)
        #     f.close()  
        # an order_id <--> a bim3d_json
   

