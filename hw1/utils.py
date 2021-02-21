import torch
import shutil
import logging
import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn


class Logger():
    """
    create a logger objetc.
    """
    def __init__(self, savepath):
        log_level = logging.INFO
        self.logger = logging.getLogger()
        handler = logging.FileHandler(f"{savepath}.log")
        handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(log_level)

    def info(self, text):
        self.logger.info(text)


def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    """
    save model and optimizer's statuses while training.
    It will also save the best model(highest testing accuracy) be seen so far.
    """
    torch.save(state, f'{path}/{filename}')
    if is_best:
        shutil.copyfile(f'{path}/{filename}', f'{path}/best-{filename}')


def _test_cpu(net, testloader, criterion, vectorize):
    net.eval()
    test_accuracy = 0.0
    batch_size = testloader.batch_size
    Ypred = []
    for i, (images, labels) in enumerate(testloader):
        if vectorize:
            images = images.view([images.shape[0], -1])
        output = net(images)
        loss = criterion(output, labels)
        predicted = output.data.max(1)[1]
        Ypred += predicted
        accuracy = (float(predicted.eq(labels.data).sum()) / float(batch_size))
        test_accuracy += accuracy
    test_accuracy_epoch = test_accuracy / (i + 1)
    test_loss_epoch = loss.item()
    return test_loss_epoch, test_accuracy_epoch, Ypred


class twoLayerNN(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        This class is used to for MNIST and CIFAR-10 image classification
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        # reference https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html
        """
        super(twoLayerNN, self).__init__()
        # input layer is a fully connected layer
        # for single input x of shape (D_in, 1)
        # 'Linear' method creates a weight matrix W of shape (H, D_in) and
        # a bias vector b of shape (H, 1) and then outputs
        # z = Wx + b of shape (H,1)
        self.fcInput = torch.nn.Linear(D_in, H, bias=True)
        self.reluHidden = torch.nn.ReLU()
        self.fcOutput = torch.nn.Linear(H, D_out, bias=True)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        input = self.fcInput(x)
        hidden = self.reluHidden(input)
        output = self.fcOutput(hidden)
        return output


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

class CNNCIFAR10(torch.nn.Module):
    """
    Train a CNN with: conv - conv - maxpool - conv - maxpool - fc + relu - fc
    Reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """
    def __init__(self):
        super(CNNCIFAR10, self).__init__()
        # int((h + 2*padding - dilation * (kernel_size-1) - 1) / stride + 1)
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,
                                     stride=1, padding=0, dilation=1, bias=True)
        self.conv2 = torch.nn.Conv2d(32, 64, 3)
        self.conv3 = torch.nn.Conv2d(64, 128, 3)
        self.pool = torch.nn.MaxPool2d(kernel_size=3)
        self.fc1 = torch.nn.Linear(512, 128, bias=True)
        self.fc2 = torch.nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TrainAndTest():
    """
    create a trainAndTest class. 
    """
    def __init__(self, net, trainloader, testloader, criterion, optimizer, netname):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.netname = netname

    def train(self, epoch):
        self.net.train()
        # decaying learning rate
        if (epoch + 1) % 30 == 0:
            self.learningRate /= 2
            if self.logger:
                self.logger.info("=> Learning rate is updated!")
            else:
                print("=> Learning rate is updated!")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learningRate
        train_accuracy = 0.0
        for i, (images, labels) in enumerate(self.trainloader):
            if self.vectorize:
                images = images.view([images.shape[0], -1])
            if self.use_cuda:
                images, labels = images.cuda(), labels.cuda()
            self.optimizer.zero_grad()
            output = self.net.forward(images)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()
            predicted = output.data.max(1)[1]
            accuracy = (float(predicted.eq(labels.data).sum()) / float(self.batch_size))
            train_accuracy += accuracy
        train_accuracy_epoch = train_accuracy / (i + 1)
        loss_epoch = loss.item()
        if not self.logger:
            print("=> Epoch: [{:4d}/{:4d}] | Training Loss:[{:2.4e}] | Training Accuracy: [{:5.4f}]".format(
                epoch + 1, self.total_epochs, loss_epoch, train_accuracy_epoch))
        else:
            self.logger.info("=> Epoch: [{:4d}/{:4d}] | Training Loss:[{:2.4e}] | Training Accuracy: [{:5.4f}]".format(
                epoch + 1, self.total_epochs, loss_epoch, train_accuracy_epoch))
        return loss_epoch, train_accuracy_epoch

    def test(self, epoch):
        self.net.eval()
        test_accuracy = 0.0
        for i, (images, labels) in enumerate(self.testloader):
            if self.vectorize:
                images = images.view([images.shape[0], -1])
            if self.use_cuda:
                images, labels = images.cuda(), labels.cuda()
            output = self.net(images)
            loss = self.criterion(output, labels)
            predicted = output.data.max(1)[1]
            accuracy = (float(predicted.eq(labels.data).sum()) / float(self.batch_size))
            test_accuracy += accuracy
        test_accuracy_epoch = test_accuracy / (i + 1)
        test_loss_epoch = loss.item()
        if not self.logger:
            print("=> Epoch: [{:4d}/{:4d}] | Testing  Loss:[{:2.4e}] | Testing  Accuracy: [{:5.4f}]".format(
                epoch + 1, self.total_epochs, test_loss_epoch, test_accuracy_epoch))
        else:
            self.logger.info("=> Epoch: [{:4d}/{:4d}] | Testing  Loss:[{:2.4e}] | Testing  Accuracy: [{:5.4f}]".format(
                epoch + 1, self.total_epochs, test_loss_epoch, test_accuracy_epoch))
        return test_loss_epoch, test_accuracy_epoch

    def build(self, total_epochs, checkpointPath=None, logger=None, modelDir="./", vectorize=False, start_epoch=0, sanitycheck=True):
        self.total_epochs = total_epochs
        self.checkpointPath = checkpointPath
        self.logger = logger
        self.modelDir = modelDir
        self.batch_size = self.trainloader.batch_size
        self.vectorize = vectorize
        self.use_cuda = torch.cuda.is_available()
        self.sanitycheck = sanitycheck
        training_accuracy_seq = []
        training_loss_seq = []
        testing_accuracy_seq = []
        testing_loss_seq = []
        testing_best_accuracy = -1
        if self.logger:
            self.logger.info(f"Number of GPU available: {torch.cuda.device_count()}")
        else:
            print(f"Number of GPU available: {torch.cuda.device_count()}")
        # try GPU
        if self.use_cuda:
            self.net.cuda()
            if not isinstance(self.net, torch.nn.DataParallel):
                self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        # start from checkpoint
        if self.checkpointPath:
            if not self.logger:
                print("Resume training from the checkpoint...")
            else:
                self.logger.info("Resume training from the checkpoint...")
            if os.path.isfile(self.checkpointPath):
                print("=> loading checkpoint from '{}'".format(self.checkpointPath))
                checkpoint = torch.load(self.checkpointPath)
                start_epoch = checkpoint['epoch'] + 1
                self.net.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                training_accuracy_seq = checkpoint['training_accuracy_seq']
                training_loss_seq = checkpoint['training_loss_seq']
                testing_accuracy_seq = checkpoint['testing_accuracy_seq']
                testing_loss_seq = checkpoint['testing_loss_seq']
                testing_best_accuracy = checkpoint['testing_best_accuracy']
                if not self.logger:
                    print("=> load checkpoint from '{}' (epochs {} are trained)"
                          .format(self.checkpointPath, start_epoch))
                else:
                    self.logger.info("=> loaded checkpoint '{}' (epoch {} are trained)"
                                     .format(self.checkpointPath, start_epoch))
                # sanitycheck
                if sanitycheck:
                    test_loss, test_accuracy = self.test(start_epoch - 1)
                    print(f'For the loaded net: testing loss: {test_loss:5.4f} | testing accuracy:[{test_accuracy:5.4f}]')
                    print(f'Recorded          : testing loss: {testing_loss_seq[-1]:5.4f} | testing accuracy:[{testing_accuracy_seq[-1]:5.4f}]')
                    assert np.abs(test_loss - testing_loss_seq[-1]) <= 1e-8, 'loading the wrong checkpoint!'

            else:
                if not self.logger:
                    print("=> no checkpoint found at '{}'".format(self.checkpointPath))
                    print("=> Training the network from scratch...")
                else:
                    self.logger.info("=> no checkpoint found at '{}'".format(self.checkpointPath))
                    self.logger.info("=> Training the resnet from scratch...")
        else:
            if not self.logger:
                print("=> Training the network from scratch...")
            else:
                self.logger.info("=> Training the resnet from scratch...")

        # get up-to-date learning rate; resume training
        self.learningRate = self.optimizer.param_groups[0]['lr']

        for epoch in range(start_epoch, self.total_epochs):
            train_loss, train_accuracy = self.train(epoch)
            test_loss, test_accuracy = self.test(epoch)
            training_loss_seq.append(train_loss)
            training_accuracy_seq.append(train_accuracy)
            testing_loss_seq.append(test_loss)
            testing_accuracy_seq.append(test_accuracy)

            is_best = testing_accuracy_seq[-1] > testing_best_accuracy
            testing_best_accuracy = max(testing_best_accuracy, testing_accuracy_seq[-1])

            state = {
                "epoch": epoch,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_loss_seq": training_loss_seq,
                "training_accuracy_seq": training_accuracy_seq,
                "testing_loss_seq": testing_loss_seq,
                "testing_accuracy_seq": testing_accuracy_seq,
                "testing_best_accuracy": testing_best_accuracy
            }
            save_checkpoint(state, is_best, path=modelDir, filename=f'{self.netname}-checkpoint.pth.tar')
            if is_best:
                if self.logger:
                    self.logger.info("=> Best parameters are updated")
                else:
                    print("=> Best parameters are updated")

        if self.logger:
            self.logger.info("Trained on [{}] epoch, with test accuracy [{}].\n \
             => During the training stages, historical best test accuracy is \
         [{}]".format(self.total_epochs, testing_accuracy_seq[-1], testing_best_accuracy))
        else:
            print("=> Trained on [{:4d}] epochs, with test accuracy [{:5.4f}].\n"
                  "=> During the training stages, historical best test accuracy is [{:5.4f}]".format(self.total_epochs,
                                                                                                     testing_accuracy_seq[-1], testing_best_accuracy))
