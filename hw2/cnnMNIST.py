import torch
import torch.nn.functional as F

class CNNMNIST(torch.nn.Module):
    """
    Train a CNN with: conv - maxpool - conv - maxpool - fc + relu - fc
    Reference: https://github.com/pytorch/examples/blob/master/mnist/main.py
    """

    def __init__(self):
        super(CNNMNIST, self).__init__()
        # int((h + 2*padding - dilation * (kernel_size-1) - 1) / stride + 1)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,
                                     stride=1, padding=0, dilation=1, bias=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=3)
        self.conv2 = torch.nn.Conv2d(32, 64, 3)
        self.fc1 = torch.nn.Linear(256, 128, bias=True)
        self.fc2 = torch.nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x