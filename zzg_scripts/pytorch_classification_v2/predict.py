"""
code by zzg 2020-07-01
predict single image
"""
##test
###预测2类示例

import torch
import os
import cv2
import glob
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from utils import settings
import torchvision.transforms as transforms
import torchvision.models as models
from utils.license_extract import extract_license, processModel


mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]

def get_test_transform():
    return transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

# img = Image.open(img_path).convert('RGB')
# img = get_test_transform()(img).unsqueeze(0)

def predict(net, process_model, imgs):

    pred_list, _id = [], []
    print("-----starting predicting!------------")
    for i in tqdm(range(len(imgs))):
        img_path = imgs[i].strip()
        print('+++', img_path)
        _id.append(os.path.basename(img_path).split('.')[0])
        img = cv2.imread(img_path)
        img1 = extract_license(process_model, img)

        if img1 is not None:
            img1 = Image.fromarray(img1.astype('uint8')).convert('RGB') 
            img1 = get_test_transform()(img1).unsqueeze(0)
            img = img1.to(device)

            with torch.no_grad():
                out = net(img)
                print(out)
            prediction = torch.argmax(out, dim=1).cpu().item()
            pred_list.append(prediction)
    return _id, pred_list


if __name__ == "__main__":

    test_model = "checkpoint/resnet50_1114/2022-11-14T15:33:32.677322/resnet50_1114-52-best-0.9001030325889587.pth"
    model_name = "resnet50"
    classname = { 0:'false', 1:'true'}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read model
    resnet50 = models.resnet50()
    resnet50.fc = torch.nn.Linear(2048, 2)
    net = resnet50
    #load_checkpoint
    net.load_state_dict(torch.load(test_model))
    net = net.to(device)
    net.eval()
    ## license extract
    process_model = processModel()

    img_Lists = glob.glob(settings.TSET_IMAGE + '/*.jpg')
    # print(img_Lists)
    _id, pred_list = predict(net, process_model, img_Lists)
    # print(classname)
    print(_id, pred_list)
    for i in range(len(pred_list)):
        print("{} --> {}".format(_id[i],classname[pred_list[i]]))

    submission = pd.DataFrame({"ID": _id, "Label": pred_list})
    submission.to_csv(settings.BASE + '{}_submission.csv'
                      .format(model_name), index=False, header=False)
