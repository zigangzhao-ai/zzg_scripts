#!/usr/bin
# Author       : zzg 
# Last modified: 2023-03-31 16:12

"""
functions: 移动补充打标注材料 -> 另一个文件夹
"""

import os
import glob

scr_dir = "0330_elec_licesne"
dst_dir = "elec_license"

json_files = glob.glob(scr_dir + '/*.json')
cnt = 0
for json_pth in json_files:
    jpg_pth = json_pth.replace(".json", ".jpg")
    if os.path.exists(jpg_pth):
        cnt  += 1
        cmd1 = "mv {} {}".format(json_pth, dst_dir)
        cmd2 = "mv {} {}".format(jpg_pth, dst_dir)
        print("---mv---", cnt, jpg_pth)
        os.system(cmd1)
        os.system(cmd2)