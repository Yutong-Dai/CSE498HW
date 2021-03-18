from matplotlib import  pyplot as plt
from utils import *
from resnetCIFAR10 import ResNet18
import random
from attacks import *

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

def vis(net, images, adv_images, labels, figtype='gray', 
                    inv_normalize= None, M = 5, start=50, savename=None, figDir=None):
    """
        create fgsm attach on given images;
        the key parameter is epsilon, which controls the strength of attack
    """
    if not inv_normalize:
        inv_normalize = transforms.Normalize(
                        mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
                        std=[1/0.2023, 1/0.1994, 1/0.2010])
    images_adv = adv_images
    pred_clean = net(images).data.max(1)[1]
    pred_adv = net(images_adv).data.max(1)[1]
    f,ax = plt.subplots(M,2, sharex=True, sharey=True, figsize=(2,M*1.3))
    for i in range(M):
        for j in range(2):
            if j == 0:
                if figtype == 'gray':
                    ax[i][0].imshow(images[i+start][0].cpu().numpy(), cmap="gray")
                else:
                    ax[i][0].imshow(tensorToImg(images[i+start], inv_normalize))
                title = ax[i][j].set_title(f"{labels[i+start]} -> {pred_clean[i+start]}")
                plt.setp(title, color=('g' if pred_clean[i+start] == labels[i+start] else 'r'))
            else:
                if figtype == 'gray':
                    ax[i][1].imshow(images_adv[i+start][0].cpu().numpy(), cmap="gray")
                else:
                    ax[i][1].imshow(tensorToImg(images_adv[i+start], inv_normalize))
                title = ax[i][j].set_title(f"{labels[i+start]} -> {pred_adv[i+start]}")
                plt.setp(title, color=('g' if pred_adv[i+start] == labels[i+start] else 'r'))
            ax[i][j].set_axis_off()
    plt.tight_layout()
    if savename is not None:
        plt.savefig(f'{figDir}/{savename}.png')
        plt.close()


testset = torchvision.datasets.CIFAR10(root=dataDir, train=False, download=False, transform=simba_utils.CIFAR_TRANSFORM)
nTestSamples, width, height, channel = testset.data.shape
image_size = height
print(f'# test samples:{nTestSamples}')
print(f'per image size: {width}*{height} | per image channel:{channel}')

net = ResNet18()
net.to(location)
net = torch.nn.DataParallel(net)
criterion = torch.nn.CrossEntropyLoss()
netname=f'cifar10-resnet'
modelPath = modelDir+ '/best-{}-checkpoint.pth.tar'.format(netname)
checkpoint = torch.load(modelPath, map_location=torch.device(location))
net.load_state_dict(checkpoint['state_dict'])


result = simba_cifa10(net, testset, num_imgs=32, batch_size=16, image_size=32, order = 'rand', 
                 num_iters = -1, targeted = False, stride = 7, epsilon = 0.2, 
                 linf_bound = 0.0, pixel_attack = True, log_every = 500, return_raw=False)