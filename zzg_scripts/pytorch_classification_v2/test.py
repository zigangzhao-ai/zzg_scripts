#test.py
#!/usr/bin/env python3

"""
test neuron network performace
print top1 and top5 err on test dataset of a model

add precision and recall as evaluation index
code by zzg 2020-06-11
"""

import argparse
from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
from models.convnext import convnext_tiny
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

from utils import settings
from utils.utils import get_test_dataloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default="convnext", help='net type')
    parser.add_argument('--weights', type=str, default="/code/zzg/project/seal_project/license_classification/checkpoint/convnext_1213/2022-12-13T20:52:52.014792/convnext_1213-55-best-0.9268.pth", help='the weights file you want to test')
    parser.add_argument('--gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('--num', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size for dataloader')
    parser.add_argument('--shuffle', type=bool, default=False, help='whether shuffle the dataset')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "resnet" in args.weights:
        print("====resnet")
        resnet50 = models.resnet50()
        resnet50.fc = torch.nn.Linear(2048, 2)
        net = resnet50

    if "convnext" in args.weights:
        convnext_tiny = convnext_tiny()
        # checkpoint = torch.load("models/pretrained_model/convnext_tiny_1k_224_ema.pth")
        # convnext_tiny.load_state_dict(checkpoint["model"], strict=False)
        convnext_tiny.head = torch.nn.Linear(768, 2)
        net = convnext_tiny

    # net = get_network(args)
    test_loader = get_test_dataloader(
        num_workers=args.num,
        batch_size=args.batch_size,
        shuffle=args.shuffle
    )

    net.load_state_dict(torch.load(args.weights))
    # print(net)
    net = net.to(device)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    # Initialize the prediction and label lists(tensors)
    pred_list = torch.zeros(0, dtype=torch.long, device='cpu')
    gt_list = torch.zeros(0, dtype=torch.long, device='cpu')

    label_names = ["false", "true"]
    num_class = len(label_names)

    for n_iter, (image, label) in enumerate(test_loader):
        # print("iteration: {}\t total {} iterations".format(n_iter + 1, len(test_loader)))
        image = image.to(device)
        label0 = label.to(device)
        output = net(image)
        # print(output)
        # print(label0)
        _, pred = output.topk(2, 1, largest=True, sorted=True)
        # print(pred)

        label = label0.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()
        #compute top 5
        correct_5 += correct[:, :2].sum()
        #compute top1 
        correct_1 += correct[:, :1].sum()

        _, preds = torch.max(output, 1)
        print(preds)
        pred_list = torch.cat([pred_list, preds.view(-1).cpu()])
        gt_list = torch.cat([gt_list, label0.view(-1).cpu()])

    report = classification_report(gt_list.numpy(), pred_list.numpy(), target_names=label_names, digits=2)
  
    print("Top 1 err: {:.3}%".format((1 - correct_1 / len(test_loader.dataset))*100))
    print("Top 5 err: {:.3}%".format((1 - correct_5 / len(test_loader.dataset))*100))
    print(report)

    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
