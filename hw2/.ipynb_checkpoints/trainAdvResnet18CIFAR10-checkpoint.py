import pickle
import os.path
import argparse
from matplotlib import  pyplot as plt
import utils
from attacks import *
from resnetCIFAR10 import ResNet18
import random
from AdvTrainAndTest import *

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn


dataDir = f'../db'
modelDir = f'./model'
logDir = f'./log'
figDir = f'./fig'
use_cuda = torch.cuda.is_available()
if use_cuda:
    location = 'cuda'
else:
    location = 'cpu'
print(location)

torch.manual_seed(1) 
random.seed(1) 
# reference for RandomCrop and RandomHorizontalFlip
# https://stackoverflow.com/questions/51677788/data-augmentation-in-pytorch
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(dataDir, train=True,  download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(dataDir, train=False,  download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)
nTrainSamples, width, height, channel = trainset.data.shape
nTestSamples, width, height, channel = testset.data.shape
print(f'# train samples: {nTrainSamples} | # test samples:{nTestSamples}')
print(f'per image size: {width}*{height} | per image channel:{channel}')

net = ResNet18()
netname=f'cifar10-resnet18-adv-pgd-0.3-0.01-25'
# choose optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
logFilePath= f'{logDir}/{netname}'
logger = utils.Logger(logFilePath)
criterion = torch.nn.CrossEntropyLoss()
checkpointPath = f'{modelDir}/{netname}-checkpoint.pth.tar'
netclf = AdvTrainAndTest(net, trainloader, testloader, 
                           criterion, optimizer, netname=netname)
attacker = ATTACKER(type='PGD', epsilon=0.3, alpha=1e-2, num_iter=25)                           
netclf.build(start_epoch=0, total_epochs=200, attacker=attacker, checkpointPath=checkpointPath, 
                           logger=logger, modelDir=modelDir)