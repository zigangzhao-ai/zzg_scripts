#!/usr/bin
# Author       : zzg 
# Last modified: 2023-03-09 11:31

import os 
import cv2
import glob
import json
import numpy as np

##step1: 整合分文件夹下图片和标签到一个文件夹
img_folder = "wait_for_check"
out_dir = "0403_all"
os.makedirs(out_dir, exist_ok=True)

for root, _, files in os.walk(img_folder):
    if len(files) > 1:
        for file in files:
            print("---", file)
            if file.endswith("jpg") or file.endswith("json"):
                file_pth = os.path.join(root, file)
                cmd = "cp -r {} {}".format(file_pth, out_dir)
                ##需要时把下面语句注释去掉
                # os.system(cmd)

##step2: 按匹配对整理，去除没有标签的图片
img_pths = glob.glob(out_dir + "/*.jpg")
for img_pth in img_pths:
    json_pth = img_pth.replace(".jpg", ".json")
    if not os.path.exists(json_pth):
        cmd = "rm -rf {}".format(img_pth)
        print("---del---", img_pth)
        ##需要时把下面语句注释去掉
        # os.system(cmd)


##step3: 将labelme_json标注转yolov5 txt
def convert(size, box):
    """
    convert [xmin, xmax, ymin, ymax] to [x_centre, y_centre, w, h]
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

out_dir = "0403_all"
txt_dir = "txt"
os.makedirs(txt_dir, exist_ok=True)

class_names = ["***"]

json_pths = glob.glob(out_dir + "/*.json")

for json_pth in json_pths:
    f1 = open(json_pth, "r")
    json_data = json.load(f1)

    img_pth = os.path.join(out_dir, json_pth.replace("json", "jpg"))
    img = cv2.imread(img_pth)
    h, w = img.shape[:2]

    tag = os.path.basename(json_pth)
    out_file = open(os.path.join(txt_dir, tag.replace("json", "txt")), "w")
    # print(json_data)
    label_infos = json_data["shapes"]
    for label_info in label_infos:
        label = label_info["label"]
        points = label_info["points"]
        print("+++", len(points))
        if len(points) >= 3:
            points = np.array(points)
            print(points.shape)
            xmin, xmax = max(0, min(np.unique(points[:, 0]))), min(w, max(np.unique(points[:, 0])))
            ymin, ymax = max(0, min(np.unique(points[:, 1]))), min(h, max(np.unique(points[:, 1])))
            print("++++", ymin, ymax)
        elif len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
        else:
            continue
        bbox = [xmin, xmax, ymin, ymax]
        bbox_ = convert((w,h), bbox)
        cls_id = class_names.index(label)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bbox_]) + '\n')

