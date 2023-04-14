#!/usr/bin
# Author       : zzg 
# Last modified: 2023-03-09 10:24

"""
functions: 转换label_json中的imageData 到 img_np，并保存为图片
"""
import numpy as np
import base64
import cv2
import os
import json
import glob


def bs2im(bs):
    img_data= base64.b64decode(bs)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

src_dir = "***"
json_pths = glob.glob(src_dir + "/*.json")
# print(json_pths)
for json_pth in json_pths:
    f = open(json_pth, "r")
    json_data = json.load(f)
    # print(json_data)
    img_data = json_data["imageData"]
    # print(img_data)
    img_np = bs2im(img_data)
    tag = os.path.basename(json_pth).replace(".json", ".jpg")
    print("===", tag)
    cv2.imwrite(os.path.join(src_dir, tag), img_np)

