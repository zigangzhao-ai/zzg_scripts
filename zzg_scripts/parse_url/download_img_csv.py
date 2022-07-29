
# -*- coding: utf-8 -*-
"""
#Time    : 2022/5/25 下午2:51
#Author  : hnsywangxin
#Contcat : hnsywangxin@gmail.com
#Descrip : 从圣都的库表中下载csv，然后根据csv下载相应的图片
"""

import json
import numpy as np
import os
import cv2
import requests as req
from concurrent.futures import ThreadPoolExecutor
import csv
import traceback

decode_url = 'http://preview.i.utopia-smart-construction-site.home.ke.com/constructionsite/camera/get/preSigned/image/url?'


csv_path = "./BW0701-0725.csv"
save_dir = "./BW0701-0725-images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
csv_data = open(csv_path, 'r')
csv_data =csv.reader(csv_data)

def download_func(img_url, order_id):
    url_params = {'originUrl': img_url}
    resp = req.get(decode_url, params=url_params)

    cons_res = json.loads(resp.content)
    if cons_res['code'] == 2000:  # and pre_cons_res['code']==2000:
        try:
            new_img_url = cons_res['data']
            # print('--', img_url)
            response = req.get(new_img_url)
            img_src = cv2.imdecode(np.frombuffer(response.content, np.uint8), flags=cv2.IMREAD_COLOR)
            cur_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
            cur_value = np.mean(cur_gray)
            #if cur_value < 35.0:
            #    return
            img_src = cv2.resize(img_src, (1408, 1080))
            cv2.imwrite(save_dir + "/" + order_id + '_' + img_url.split("/")[-1], img_src)
            print("download {}".format(img_url.split("/")[-1]))
        except:
            traceback.print_exc()

pool = ThreadPoolExecutor(8)
i = 0
for line in csv_data:
    if csv_data.line_num == 1:  # 过滤第一行
        continue
    i += 1
    #if i > 20:
    #    break
    img_url = line[3]
    order_id = line[0]
    #pool.submit(download_func, img_url, type_flag,normal_path, extra_path)
    pool.submit(download_func, img_url, order_id)
