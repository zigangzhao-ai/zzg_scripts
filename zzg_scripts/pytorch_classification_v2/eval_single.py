import torch
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F

class sealClassification():
    
    def __init__(self, weights_resnet50, opt_device):

        ## resnet50 for seal classification
        seal_class_net = models.resnet50(pretrained=False, num_classes=2)
        seal_class_net.load_state_dict(torch.load(weights_resnet50, map_location=opt_device)) #map_location=opt_device
        self.device = opt_device
        self.seal_class_net = seal_class_net.to(self.device).eval()

        ## config
        self.lable_dict = {0:'false', 1:'true'}
        self.flag_dict = {0: "假章", 1: "真章", 2: "疑似真假", 3:"未检测到印章"}
    
    def get_test_transform(self):
        '''
        data_process for seal classification
        '''
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        return transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def judge_seal_class_(self, img_seal):

        seal_img_rotate = Image.fromarray(img_seal.astype('uint8')).convert('RGB')
        seal_img_rotate = self.get_test_transform()(seal_img_rotate).unsqueeze(0).to(self.device)

        out = self.seal_class_net(seal_img_rotate)
        out = F.softmax(out, dim=1) ## x.shape=[1,2]按行计算
        label = torch.argmax(out, dim=1).cpu().item()
        conf = out[0][label].cpu().item()

        if label == 0:
            flag = 0
           
        if label == 1:
            flag = 1 
        # print('-----', conf, self.flag_dict[flag])  
        return flag, conf
    


if __name__ == '__main__':
    
    import glob
    import cv2

    img_true_list = glob.glob("/code/zzg/project/seal_project/seal_classification/dataset/test_data/1009_test/佛山市禅城区市场监督管理局/true/*.jpg")
    img_false_list = glob.glob("/code/zzg/project/seal_project/seal_classification/dataset/test_data/1009_test/成都市温江区市场监督管理局/false/*.jpg")

    opt_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  ## device = 'cpu' or '0,1,2,3'
    weights_resnet50 = "/code/zzg/project/seal_project/seal_classification/checkpoint/resnet50_1017/resnet50-0.991-北京_all-1017.pth"
    weights_seal_rect = r'common_models/yolov5s_sealRotate_640_0913.pt'


    flag_dict = {0: "假章", 1: "真章", 2: "疑似真假", 3:"未检测到印章"}

    seal_classification = sealClassification(weights_resnet50, weights_seal_rect, opt_device)

    flag = 'true'
    if flag == 'true':
        test_list = img_true_list
        
    else:
        test_list = img_false_list

    cnt_true = 0
    cnt_false = 0
    for img_pth in test_list:

        seal_img = cv2.imread(img_pth)
        flag, conf = seal_classification.judge_seal_class_(seal_img)
        print('-----', img_pth, conf, flag_dict[flag])
        if flag == 1:
            cnt_true += 1
        else:
            cnt_false += 1
    
print(len(test_list), cnt_true)
print(len(test_list), cnt_false)
            


        


    