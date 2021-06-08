#test.py
#!/usr/bin/env python3
"""
test neuron network performace
print top1 and top5 err on test dataset of a model

add precision and recall as evaluation index
code by mvp @12.12.26
"""

import argparse
import torch
import os
import torch.nn as nn
import torchvision
import numpy as np
import torchvision.models as models
from sklearn.metrics import classification_report
from net.resnet import resnet50
from utils import get_test_dataloader, label_names, class_accuracy
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str, default='model', help='net type')
    parser.add_argument('--basenet', type=str, default='model_best.pth.tar', help='net type')
    parser.add_argument('--gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('--num', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('--shuffle', type=bool, default=False, help='whether shuffle the dataset')
    args = parser.parse_args()

    ## load test_data
    test_loader = get_test_dataloader(
        num_workers=args.num,
        batch_size=args.batch_size,
        shuffle=args.shuffle
    )

    # resnet50 = models.resnet50()
    resnet50 = resnet50()
    resnet50.cls = torch.nn.Linear(2048, 98)
    net = resnet50

    ##load weights
    resnet50_weights = torch.load(args.model_folder + '/'+ args.basenet, map_location='cpu')
    #map_location={'cuda:2':'cuda:0'}) 
    #map_location=lambda storage, loc: storage)
    net.load_state_dict(resnet50_weights["state_dict"])
   
   
    ###origi Acc@1 
    best_acc1 = resnet50_weights["best_acc1"]
    print("Acc@1 in val :{}".format(best_acc1))


    print("---------start testing----------------")
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    # Initialize the prediction and label lists(tensors)
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')

    class_correct = np.array([0] * 98)
    total = np.array([0] * 98)
    
    n = 0
    for _, (image, label) in enumerate(test_loader):
        
        n += 1
        print(n)
        if args.gpu:
            image = image.cuda()
            label0 = label.cuda()            
            net = net.cuda()
        else:
            image = image
            label0 = label
            net = net

        output = net(image)

        class_correct, total = class_accuracy(output, label0, class_correct, total)
        
        _, pred = output.topk(5, 1, largest=True, sorted=True)
  
        label = label0.view(label.size(0), -1).expand_as(pred)
        # print(pred,label)
        correct = pred.eq(label).float()

        #compute top 5
        correct_5 += correct[:, :5].sum()
        #compute top1 
        correct_1 += correct[:, :1].sum()

        _, preds = torch.max(output, 1)

        predlist = torch.cat([predlist, preds.view(-1).cpu()])
        lbllist = torch.cat([lbllist, label0.view(-1).cpu()])

    accuracy_list = 1.0 * class_correct / total
    report = classification_report(lbllist.numpy(), predlist.numpy(), target_names=label_names, digits=3)

    # print(predlist.numpy(), lbllist.numpy())
    print("Top 1 err: {:.3}%".format((1 - correct_1 / len(test_loader.dataset))*100))
    print("Top 5 err: {:.3}%".format((1 - correct_5 / len(test_loader.dataset))*100))
    print("p-r of each class:", report)
    print("acc of each class:", accuracy_list)
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))


    # TEST_DATA_PATH = "val"
    # test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH)
    # print(test_data.class_to_idx)
