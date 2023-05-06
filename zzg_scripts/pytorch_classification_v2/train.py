
# train.py
#!/usr/bin/env	python3

""" 
coede by zzg 2020-06-11
training by resnet50

"""
import os
import re
import sys
import argparse
from datetime import datetime
from torch.backends import cudnn

import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import settings
from models.convnext import convnext_tiny
from torch.hub import load_state_dict_from_url
from utils.utils import  get_training_dataloader, get_test_dataloader, WarmUpLR
from utils.data_augmentation import mixup_data, mixup_criterion, LabelSmoothCEloss, cutmix


##set random seed
seed = 3
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed)    # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(epoch):
    
    train_loss = 0.0 # cost function error
    correct = 0.0
    ##use mixup
    ismixup = False
    iscutmix = False
    r = np.random.rand(1)

    for batch_index, (images, labels) in enumerate(training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()
        # print(images, labels)
        if ismixup:
            inputs, targets_a, targets_b, lam = mixup_data(images.cuda(), labels.cuda(), alpha=1.0)

            optimizer.zero_grad()
            outputs = net(inputs)            
            _, preds = outputs.max(1)
            correct += lam * preds.eq(targets_a).sum() + (1-lam) * preds.eq(targets_b).sum()
            
            loss = mixup_criterion(loss_function, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

        if r < args.cutmix_prob and iscutmix:
            inputs, targets_a, targets_b, lam = cutmix(images.cuda(), labels.cuda(), alpha=1.0)
            
            optimizer.zero_grad()
            outputs = net(inputs)
                       
            _, preds = outputs.max(1)
            correct += lam * preds.eq(targets_a).sum() + (1-lam) * preds.eq(targets_b).sum()
            
            loss = mixup_criterion(loss_function, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

        else:
            labels = labels.to(device)
            images = images.to(device)

            optimizer.zero_grad()
            outputs = net(images)

            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

            loss = loss_function(outputs, labels)  ## baseline
            loss.backward()
            optimizer.step()


        print('Training Epoch: [ {epoch} || [{trained_samples}/{total_samples}]\t || Loss: {:0.4f}\t || LR: {:0.6f} ]'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.batch_size + len(images),
            total_samples=len(training_loader.dataset)
        ))

        train_loss += loss.item()
    
    training_loss = train_loss / len(training_loader.dataset)
    training_acc = float(correct.float() / len(training_loader.dataset))
    loss_train.append(training_loss)
    acc_train.append(training_acc)
    #print(loss_train,acc_train)
    print('[train set: Average loss: {:.4f} || Accuracy: {:.4f}]'.format(
        training_loss, training_acc))
  

def eval_training():
    
    net.eval()
    test_loss = 0.0 # cost function error
    correct = 0.0
 
    for (images, labels) in test_loader:
       
        images = images.to(device)
        labels = labels.to(device)
        # net = net.cuda()
        with torch.no_grad():
            outputs = net(images)
            loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        # print(correct)      
    print('[Test set: Average loss: {:.4f} || Accuracy: {:.4f}]'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset)
    ))
  
    loss_test.append(test_loss / len(test_loader.dataset))
    acc_test.append(float(correct.float() / len(test_loader.dataset)))
    #print(loss_test,acc_test)
    return correct.float() / len(test_loader.dataset)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='convnext_1213', help='net type') #required=True,
    parser.add_argument('--numworks', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for dataloader') #default=32
    parser.add_argument('--shuffle', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('--warm', type=int, default=5, help='warm up training phase')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--cutmix-prob', default=0.5, type=float, help='cutmix probability')
    parser.add_argument('--auto-augment', default=True, type=bool, help='data auto augment')
    parser.add_argument('--cutout', default=True, type=bool, help='data augment-cutout')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ##load pretrain model
    if "resnet" in args.net:
        resnet50 = models.resnet50(pretrained=True)
        resnet50.fc = torch.nn.Linear(2048, 2)
        net = resnet50
    
    if "convnext" in args.net:
        convnext_tiny = convnext_tiny() #num_classes=21841
        checkpoint = torch.load("models/pretrained_model/convnext_tiny_1k_224_ema.pth")
        convnext_tiny.load_state_dict(checkpoint["model"], strict=False)
        convnext_tiny.head = torch.nn.Linear(768, 2)
        net = convnext_tiny

    net = net.to(device)
    # print("load success!!")
    # data preprocessing:
    training_loader = get_training_dataloader(
        num_workers=args.numworks,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        auto_augment=args.auto_augment,
        cutout= args.cutout
        )
    test_loader = get_test_dataloader(
        num_workers=args.numworks,
        batch_size=args.batch_size,
        shuffle=args.shuffle
        )
    
    # loss_function = nn.CrossEntropyLoss()
    loss_function = LabelSmoothCEloss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #### warm_up_with_cosine_lr
    warm_up_with_cosine_lr = lambda epoch: epoch / args.warm if epoch <= args.warm else 0.5 * ( math.cos((epoch - args.warm) /(settings.EPOCH - args.warm) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
   
    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    print("start training!!")

    loss_train = []
    acc_train = []
    loss_test = []
    acc_test = []
    
    # acc = eval_training(1)
    for epoch in range(1, settings.EPOCH+1):
        if epoch > args.warm:
            scheduler.step(epoch)

        train(epoch)
        # eval_training(epoch)
        acc = eval_training()

        #start to save best performance model after learning rate decay to 0.01 
        if epoch >= settings.MILESTONES[0] and best_acc < acc:
            best_acc = round(float(acc), 5)
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best-{}'.format(best_acc)))
           # torch.save(net, checkpoint_path.format(net=args.net, epoch=epoch, type='best-{}'.format(best_acc)))          
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular-{}'.format(acc)))
            #torch.save(net, checkpoint_path.format(net=args.net, epoch=epoch, type='regular-{}'.format(acc)))

# import matplotlib.pyplot as plt
# import numpy as np

# loss_train = [0.005489437063086382, 0.0004661825320837255, 0.00013794747702444954, 8.897417378609319e-05, 6.640919799161856e-05, 5.271508172866784e-05, 5.453150360511748e-05, 3.6368893013028604e-05, 3.3792670152333784e-05, 2.044788483317254e-05, 1.2546866328919048e-05, 1.1675171567657306e-05, 1.3222659281990096e-05, 9.308248279849279e-06, 8.609949444401808e-06, 9.248599371386235e-06, 9.974492759603422e-06, 7.739522064586706e-06, 8.945144201384457e-06, 5.442028590621323e-06, 5.6335201619748935e-06, 8.514744099524695e-06, 8.602486083772834e-06, 9.46496205753456e-06, 7.197290911328877e-06, 7.985389834552111e-06, 7.489400347102819e-06, 8.88740328725538e-06, 8.993345007096507e-06, 6.1416200750041495e-06, 8.760668280356526e-06, 4.308918545111143e-06, 4.752806418880134e-06, 3.4718836420759262e-06, 8.589874015433436e-06, 6.382848624431073e-06, 5.700201906387303e-06, 5.388043250463207e-06, 6.0345621366054495e-06, 6.721696757530497e-06, 4.992587072087852e-06, 5.435692300340573e-06, 5.226216357499171e-06, 5.229206367692125e-06, 8.214287394720487e-06, 4.952935072144097e-06, 4.1111178145435384e-06, 6.204691015138471e-06, 5.552364590280457e-06]
# acc_train = [0.5807321071624756, 0.9617555737495422, 0.9896193146705627, 0.9931159615516663, 0.9950827956199646, 0.9963212013244629, 0.9958477020263672, 0.9973410367965698, 0.9969404339790344, 0.9986886978149414, 0.9990893602371216, 0.9991621971130371, 0.9991257786750793, 0.9994171857833862, 0.999453604221344, 0.9993807673454285, 0.9993807673454285, 0.9995264410972595, 0.999453604221344, 0.999599277973175, 0.9997814297676086, 0.999599277973175, 0.9995628595352173, 0.9993443489074707, 0.9994900226593018, 0.9994171857833862, 0.9995264410972595, 0.9995264410972595, 0.9991986751556396, 0.999599277973175, 0.9993807673454285, 0.9997814297676086, 0.9997450113296509, 0.9999635219573975, 0.9994900226593018, 0.9997450113296509, 0.9996721744537354, 0.9996356964111328, 0.9997085928916931, 0.9996356964111328, 0.9997450113296509, 0.9997085928916931, 0.9997450113296509, 0.9997450113296509, 0.9994900226593018, 0.9996721744537354, 0.9997450113296509, 0.9996356964111328, 0.9997085928916931]

# loss_test = [0.0013841308541159621, 0.0004949015759999567, 0.00030078927997231685, 0.0001498546778046247, 0.00027346015350484875, 0.0002362435050952787, 0.00025036062980855035, 0.00014558425526369967, 0.00028985873102618207, 0.00019140651800689988, 0.00020284104647662343, 0.00015609272211848261, 0.00019749342783354743, 0.00017893792307175743, 0.0001900321778015956, 0.00016582612264059936, 0.00017095471478321335, 0.000175199660223769, 0.00018368053966296897, 0.00019395164703888993, 0.00019259498248420077, 0.00020182396405222226, 0.00018292929970945517, 0.0001550281780828528, 0.00017288966598877413, 0.00015566603327871233, 0.0001366786628637444, 0.00015537324062835819, 0.00017844747915049538, 0.00017264025982903557, 0.00016351903093240146, 0.0001557743549762461, 0.00016370155561322682, 0.00016371678400990423, 0.00016788544047081596, 0.00014774972386969845, 0.0001598901455059473, 0.00016534553681813466, 0.00016778120012695205, 0.00016061030406186983, 0.0001668102550489139, 0.00016911804667745165, 0.00017030475447156945, 0.00015100644308215603, 0.00025944279613803367, 0.00016587602761010517, 0.00017124475164331674, 0.00016714461829266068, 0.00015595789460649188] 
# acc_test = [0.8855270743370056, 0.9616564512252808, 0.9746235609054565, 0.990379273891449, 0.9796431064605713, 0.9810373783111572, 0.9802008271217346, 0.9863357543945312, 0.980758547782898, 0.9854992032051086, 0.9845231771469116, 0.9859175086021423, 0.9846625924110413, 0.985080897808075, 0.984383761882782, 0.9849414825439453, 0.9848020672798157, 0.9845231771469116, 0.9845231771469116, 0.9839654564857483, 0.9845231771469116, 0.9831288456916809, 0.9838260412216187, 0.9854992032051086, 0.9845231771469116, 0.9841048717498779, 0.9854992032051086, 0.9848020672798157, 0.9842442870140076, 0.9838260412216187, 0.9856386184692383, 0.9856386184692383, 0.9846625924110413, 0.9854992032051086, 0.9838260412216187, 0.986056923866272, 0.984383761882782, 0.9842442870140076, 0.984383761882782, 0.9845231771469116, 0.985080897808075, 0.9838260412216187, 0.9845231771469116, 0.9868935346603394, 0.985080897808075, 0.9849414825439453, 0.9849414825439453, 0.9842442870140076, 0.9849414825439453]

ax1 = plt.subplot()
ax2 = ax1.twinx() #shared x axis with each other
ax1.plot(np.arange(1, len(loss_train) + 1), loss_train, color = 'g', label = 'train loss', linestyle = '-', linewidth = 2)
ax1.plot(np.arange(1, len(loss_test ) + 1), loss_test, color = 'b', label = 'test loss', linestyle = '-', linewidth = 2)
ax2.plot(np.arange(1, len(acc_train) + 1), acc_train, color = 'g', label = 'train acc', linestyle = '-', linewidth = 2)
ax2.plot(np.arange(1, len(acc_test) + 1), acc_test, color = 'b', label = 'test acc', linestyle = '-', linewidth = 2)

ax1.legend(loc=(0.7, 0.7))  #使用(0.7,0.7)定义标签位置
ax2.legend(loc=(0.7, 0.5))
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax2.set_ylabel('accuracy')
plt.savefig("output/{}.png".format(args.net), dpi = 400)
plt.show()

