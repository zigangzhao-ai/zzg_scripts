import onnxruntime
import cv2
import torch
import time
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def softmax(x): 
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

class onnxClass():

    def __init__(self, onnx_weights, device='cpu'):
        self.weights = onnx_weights
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device != 'cpu' else ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(onnx_weights, providers=providers)
    
    def get_test_transform(self):
        return transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                                 std=[0.2023, 0.1994, 0.2010]),
            ])
    
    def onnx_infer(self, img):

        img = Image.fromarray(img.astype('uint8')).convert('RGB') 
        img = self.get_test_transform()(img).unsqueeze(0)
        img = img.detach().cpu().numpy()
        inputs = {self.session.get_inputs()[0].name: img}
        outs = self.session.run(None, inputs)[0]
        outs = softmax(outs)
        # print(outs)
        pred = np.argmax(outs)
        return pred


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    onnx_weights = "onnx/resnet18-5c106cde.onnx"
    img_pth = "**.jpg"
     
    img = cv2.imread(img_pth)

    onnx_class = onnxClass(onnx_weights)
    out = onnx_class.onnx_infer(img)

    print('--', out)

