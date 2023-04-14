#!/usr/bin
# Author       : zzg 
# Last modified: 2022-10-27 10:37

"""
检测onnx推理
"""
# -*- coding:utf-8 -*-
import onnxruntime
import cv2
import numpy as np
from img_utils import scale_coords, non_max_suppression, letterbox


class yolov5Det():
    def __init__(self, weights, img_size=(640, 640), conf_thres=0.45,  
                 iou_thres=0.50, max_det=1000, agnostic_nms=False, device='cpu'):

        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det 
        self.agnostic_nms = agnostic_nms
        self.device = device

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device != 'cpu' else ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(weights, providers=providers)
     
        self.names = ["Zhengzhao_materials", "Zhengzhao_title", "License_title", "fuben"]  ##换成自己模型对应的类名即可


    def data_preprocess(self, img0s):
        # Set Dataprocess & Run inference
        img = letterbox(img0s, new_shape=self.img_size, auto=True)[0]
        # print("===", img.shape)
        # cv2.imwrite('result/0414_auto.jpg', img)
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = img.astype(dtype=np.float32)
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        return img

    
    def pred(self, img0s):

        """
        对输入的图片进行目标检测，返回对应的类别的检测框,并可视化输出
        #输出后自己做对应的逻辑处理
        """

        img = self.data_preprocess(img0s)
        # Inference
        pred = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: img})[0]     
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, self.agnostic_nms, max_det=self.max_det)
        det = pred[0] # detections single image
        # Process detections
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0s.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{self.names[int(cls)]}'
                prob = round(float(conf), 2)  # round 2
                # c_x = (int(xyxy[0]) + int(xyxy[2])) / 2
                # c_y = (int(xyxy[1]) + int(xyxy[3])) / 2
                # Img vis
                xmin, ymin, xmax, ymax = xyxy
                newpoints = [(int(xmin), int(ymin)), (int(xmax), int(ymax))]
                self.draw_vis(img0s, newpoints, label, prob)
                print('-----', img0s.shape, xyxy, label)
        return img0s

    def draw_vis(self, img, pts, label, prob):
        # vis draw
        font = cv2.FONT_HERSHEY_SIMPLEX
        newpoints = np.array(pts)    
        cv2.rectangle(img, newpoints[0], newpoints[1], (0,255,0), 2) 
        cv2.putText(img, label+'_'+str(prob), newpoints[0], font, 1, (0,0,255), 1, cv2.LINE_AA)

        return img
 
if __name__=="__main__":
    

    import sys
    import glob
    import matplotlib.pyplot as plt

    onnx_weight_path = "runs/train/exp_0403/weights/best.onnx"
    # print('----', database)
    img_pths = glob.glob("datasets/train_0403/val/images/*.jpg")
    
    cert_material_det = yolov5Det(onnx_weight_path)
    for img_pth in img_pths[:1]:
        img1 = cv2.imread(img_pth)  

        out = cert_material_det.pred(img1)
        print('++++', out.shape)

        cv2.imwrite('result/04141.jpg', out)
        
        