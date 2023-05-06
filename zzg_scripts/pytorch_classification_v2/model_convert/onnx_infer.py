import onnxruntime
import cv2
import torch
import time
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

def softmax(x): 
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]

def get_test_transform():
    return transforms.Compose([
        transforms.Resize([224, 320]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


onnx_weights = "onnx/convnext_224_320_stitch-0.917_1122.onnx"
img_pth = "/code/zzg/project/seal_project/license_classification/dataset/train_data_1121/test/false/4428_200_200.jpg"

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device != 'cpu' else ['CPUExecutionProvider']
session = onnxruntime.InferenceSession(onnx_weights, providers=providers)
print(session.get_providers())

img = cv2.imread(img_pth)
img = Image.fromarray(img.astype('uint8')).convert('RGB') 
img = get_test_transform()(img).unsqueeze(0)
#img = img.to(device)
img = img.detach().cpu().numpy()
print("===", img.shape)
t1 = time.time()
for i in range(50):
    inputs = {session.get_inputs()[0].name: img}
    outs = session.run(None, inputs)[0]
    outs = softmax(outs)
    print(outs)
t2 = time.time()

print(t2-t1)
