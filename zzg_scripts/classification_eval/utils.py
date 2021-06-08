''''
code by mvp @12.12.26
'''
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms


TEST_DATA_PATH = "val"


label_names = ['000000051', '000000061', '000000151', '010000021', '010000031', 
        '010001011', '010002051', '010300021', '020000011', '020000021', '020000031',
        '020000091', '020000111', '020001011', '020001031', '020001051', '020100031', 
        '020100041', '020100051', '030000011', '030000031', '030000041', '030000051',
        '030000071', '030000081', '030000111', '030000121', '030100011', '030100021', 
        '030100041', '030100051', '030100111', '030200011', '030200021', '030200041', 
        '030200061', '030200121', '030200131', '030200141', '030200151', '030200162', 
        '030200172', '030300171', '030300182', '040000011', '040000051', '040000071', 
        '040001011', '040001032', '040001041', '040100011', '040102031', '040102041', 
        '040200011', '040201011', '040202011', '040204011', '040204041', '040206011',
        '040209011', '040210011', '040302011', '040302022', '040303011', '040303021', 
        '040303022', '040303031', '040303041', '040304011', '040304021', '040305021', 
        '040500011', '040500012', '040500013', '040500021', '040500022', '040500023', 
        '040500031', '040501011', '040501021', '040501022', '040501023', '040501031', 
        '040501032', '040503011', '060500021', '060600031', '060700011', '060700021', 
        '060800011', '070000011', '070000021', '070000042', '070001041', '070002011', 
        '070301051', '070400011', '070400021']

def get_test_dataloader(batch_size=16, num_workers=2, shuffle=True):
    """ return test dataloader
    Args:
        mean: mean of mvp test dataset
        std: std of mvptest dataset
        path: path to mvptest python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: mvp_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),   
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
   
    test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=transform_test)

    print(test_data.class_to_idx)
    print(test_data[0][0].shape)
    # print(test_data.imgs)
    testloader = data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return testloader


def class_accuracy(output, target, class_correct, total):
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
