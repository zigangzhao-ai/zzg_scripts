from imutils import paths
import cv2
import sys
import shutil
import os

img_dir_path = 'vehicle_color_copy/JPEGImages/yellow'
orange_dir_path ='vehicle_color_copy/JPEGImages/orange'
golden_dir_path ='vehicle_color_copy/JPEGImages/golden'
unknow_dir_path ='vehicle_color_copy/JPEGImages/unknow'

img_paths = []
img_paths += [el for el in paths.list_images(img_dir_path)]


for imgpath in img_paths:
    img_temp = cv2.imread(imgpath)
    window_name = 'image'
    cv2.imshow(window_name, img_temp)
    keycode =  cv2.waitKey(0)
    print(keycode)
    if keycode == 111:
        #orange  o
        shutil.move(imgpath, orange_dir_path) 
    elif keycode == 103:
        #golden  g
        shutil.move(imgpath, golden_dir_path)
    elif keycode == 113:
        cv2.destroyAllWindows()
        exit()
        #kill    q 
    elif keycode == 110:
        #not sure n
        shutil.move(imgpath, unknow_dir_path)
# 'orange: o keycode 111'
# 'golden: g keycode 103'