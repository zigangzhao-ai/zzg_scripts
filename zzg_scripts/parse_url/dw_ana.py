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

# decode_url = 'http://preview.i.utopia-smart-construction-site.home.ke.com/constructionsite/camera/get/preSigned/image/url?'
decode_url = 'http://preview.i.utopia-smart-construction-site.home.ke.com/constructionsite/camera/get/preSigned/image/url?'

save_dir = "af_parse_img/281878"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

excel_file = '/Users/zzg_beike/Desktop/test_code/parse_url/data_csv/1025_ownfish.xlsx'
df = pd.read_excel(excel_file, sheet_name='sheet1', header=0, usecols=[0,1,2,3,4,5])

data = df.values
row = data.shape[0]
col = data.shape[1]
print(row, col)
# print(data[0, :])
cnt = 0
for i in range(0, row, 1):
    
    order_id = data[i, 1]
    room_id = data[i, 2]
    device_id = data[i, 3]
    img_url_bef = data[i, 4]
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

        cv2.imwrite(os.path.join(save_dir, str(device_id)+'_'+ room_id + '_'+ str(cnt)+'.jpg'), img_src)

