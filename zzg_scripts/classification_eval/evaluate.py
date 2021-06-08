import argparse
import glob

import torch.nn as nn
import os
import time
import torch
import numpy as np
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import utils
from train import accuracy
from utils import AverageMeter, ProgressMeter
from model import EfficientNet


mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print("Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex")
    mixed_precision = False  # not installed


def main():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Test")
    parser.add_argument(
        "-j", "--workers", default=8, type=int, metavar="N", help="number of data loading workers (default: 8)"
    )
    parser.add_argument(
        "-a", "--arch", metavar="ARCH", default="resnet18", help="model architecture (default: resnet18)"
    )
    parser.add_argument("--data", metavar="DIR", help="path to dataset")
    parser.add_argument("-b", "--batch-size", default=64, type=int, metavar="N", help="mini-batch size (default: 64)")
    parser.add_argument("--num_classes", default=2, type=int, help="the classes number")
    parser.add_argument("-p", "--print-freq", default=10, type=int, metavar="N", help="print frequency (default: 10)")
    parser.add_argument(
        "--weights", default="", type=str, metavar="PATH", help="path to test the models (default: none)"
    )
    parser.add_argument("--device", default="", help="device id (i.e. 0 or 0,1 or cpu)")
    parser.add_argument("--image_size", default=224, type=int, help="image size")
    parser.add_argument("--advprop", default=False, action="store_true", help="use advprop or not")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training. ")
    parser.add_argument("--name", default="test", help="rename the test log")

    args = parser.parse_args()
    if os.path.isfile(args.name + ".txt"):
        os.remove(args.name + ".txt")
    utils.print_savetxt(args.name, args)
    # set the device
    global mixed_precision
    device = utils.select_device(args.device, apex=mixed_precision, batch_size=args.batch_size)
    if device.type == "cpu":
        mixed_precision = False

    main_work(args, device)


def class_accuracy(output, target, class_correct, total, args):
    """Computes the accuracy of each the class"""
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target.view(1, -1))
        for i in range(len(target)):
            total[(target[i]).item()] += 1
            class_correct[(target[i]).item()] += int((correct[0, i]).item())
    return class_correct, total


def test(val_loader, model, criterion, args, device):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    class_correct = np.array([0] * args.num_classes)
    total = np.array([0] * args.num_classes)
    if args.num_classes > 5:
        top5 = AverageMeter("Acc@5", ":6.2f")
        progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5, prefix="Test: ")
    else:
        progress = ProgressMeter(len(val_loader), batch_time, losses, top1, prefix="Test: ")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            if args.num_classes > 5:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
            else:
                acc1, _ = accuracy(output, target, topk=(1, 1))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))

            class_correct, total = class_accuracy(output, target, class_correct, total, args)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        if args.num_classes > 5:
            print(" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5))
        else:
            print(" * Acc@1 {top1.avg:.3f}".format(top1=top1))

        accuracy_list = 1.0 * class_correct / total
        for index in range(args.num_classes):
            print("index : '{index}'  Acc@1 : {acc1:.3f}".format(index=index - 1, acc1=accuracy_list[index]))

    return loss, top1.avg


def main_work(args, device):
    best_acc1 = 0

    assert args.weights is not None, "The weights you must input in order to evaluate."
    if "efficientnet" in args.arch:
        print("=> creating model '{}'".format(args.arch))
        model = EfficientNet.from_name(args.arch, advprop=args.advprop, num_classes=args.num_classes)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    criterion = nn.CrossEntropyLoss().to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    testdir = os.path.join(args.data, "test")

    if args.advprop:
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if "efficientnet" in args.arch:
        image_size = EfficientNet.get_image_size(args.arch)
    else:
        image_size = args.image_size

    test_dataset = datasets.ImageFolder(
        testdir,
        transforms.Compose(
            [
                transforms.Resize(size=image_size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    print(test_dataset.class_to_idx)

    # The input of weights is path/filepath
    if os.path.isdir(args.weights):
        weights = glob.glob(os.path.join(args.weights, "*.pth"))
        for weight in weights:
            print("=> loading checkpoint '{}'".format(weight))
            checkpoint = torch.load(weight)
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            print("Acc@1 in val :{}".format(best_acc1))
            model.load_state_dict(checkpoint["state_dict"])
            print("=> loaded checkpoint '{}'".format(weight))

            model = model.to(device)
            val_loss, acc1 = test(test_loader, model, criterion, args, device)

    elif os.path.isfile(args.weights):
        print("=> loading checkpoint '{}'".format(args.weights))
        checkpoint = torch.load(args.weights)
        args.start_epoch = checkpoint["epoch"]
        best_acc1 = checkpoint["best_acc1"]
        print("Acc@1 in val :{}".format(best_acc1))
        model.load_state_dict(checkpoint["state_dict"])
        print("=> loaded checkpoint '{}'".format(args.weights))

        model = model.to(device)
        val_loss, acc1 = test(test_loader, model, criterion, args, device)

    else:
        print("=> no checkpoint found at '{}'".format(args.weights))


if __name__ == "__main__":
    main()
