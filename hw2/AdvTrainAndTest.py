import torch
import os
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import save_checkpoint
from attacks import fgsm_delta, pgd_delta

class AdvTrainAndTest():
    """
    create a trainAndTest class. 
    """

    def __init__(self, net, trainloader, testloader, criterion, optimizer, netname, scheduler=None):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.netname = netname
        if not scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        else:
            self.scheduler = scheduler
        self.use_cuda = torch.cuda.is_available()

    def one_epoch(self, epoch, type='train', attacker=None, optimizer=None):
        if type == 'train':
            self.net.train()
            loader = self.trainloader
        else:
            self.net.eval()
            loader = self.testloader
        loss, correct = 0.0, 0
        for i, (images, labels) in enumerate(loader):
            if self.use_cuda:
                images, labels = images.cuda(), labels.cuda()
            if attacker is not None:
                delta = attacker.attack(self.net,  self.criterion, images, labels)
                output = self.net.forward(images+delta)
            else:
                output = self.net.forward(images)
            loss = self.criterion(output, labels)
            if optimizer is not None:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # get accuracy
            predicted = output.data.max(1)[1]
            correct += predicted.eq(labels.data).sum().item()
            loss += loss.item() * images.shape[0]
        loss_epoch = loss / len(loader.dataset)
        accuracy_epoch = correct / len(loader.dataset)
        if attacker is not None:
            ext='adv'
        else:
            ext='   '
        if not self.logger:
            print(f"=> Epoch: [{epoch + 1:4d}/{self.total_epochs:4d}] | {type}-{ext} Loss:[{loss_epoch:2.4e}] | {type}-{ext} Accuracy: [{accuracy_epoch:5.4f}]")
        else:
            self.logger.info(f"=> Epoch: [{epoch + 1:4d}/{self.total_epochs:4d}] | {type}-{ext} Loss:[{loss_epoch:2.4e}] | {type}-{ext} Accuracy: [{accuracy_epoch:5.4f}]")
        return loss_epoch, accuracy_epoch
        

    def build(self, total_epochs, attacker=None, checkpointPath=None, logger=None, modelDir="./", start_epoch=0, sanitycheck=True):
        self.total_epochs = total_epochs
        self.checkpointPath = checkpointPath
        self.logger = logger
        self.modelDir = modelDir
        self.batch_size = self.trainloader.batch_size
        self.sanitycheck = sanitycheck
        training_accuracy_seq = []
        training_loss_seq = []
        testing_accuracy_seq = []
        testing_loss_seq = []
        adv_training_accuracy_seq = []
        adv_training_loss_seq = []
        adv_testing_accuracy_seq = []
        adv_testing_loss_seq = []
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
                adv_training_accuracy_seq = checkpoint['adv_training_accuracy_seq']
                adv_training_loss_seq = checkpoint['adv_training_loss_seq']
                adv_testing_accuracy_seq = checkpoint['adv_testing_accuracy_seq']
                adv_testing_loss_seq = checkpoint['adv_testing_loss_seq']
                if not self.logger:
                    print("=> load checkpoint from '{}' (epochs {} are trained)"
                          .format(self.checkpointPath, start_epoch))
                else:
                    self.logger.info("=> loaded checkpoint '{}' (epoch {} are trained)"
                                     .format(self.checkpointPath, start_epoch))
                # sanitycheck
                if sanitycheck:
                    test_loss, test_accuracy = self.one_epoch(start_epoch - 1, type='test', attacker=None, optimizer=None)
                    print(f'For the loaded net: testing loss: {test_loss:5.4f} | testing accuracy:[{test_accuracy:5.4f}]')
                    print(f'Recorded          : testing loss: {testing_loss_seq[-1]:5.4f} | testing accuracy:[{testing_accuracy_seq[-1]:5.4f}]')
                    assert np.abs(test_loss.detach().cpu() - testing_loss_seq[-1].detach().cpu()) <= 1e-6, 'loading the wrong checkpoint!'

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
            # normal traning
            if attacker == None:
                train_loss, train_accuracy = self.one_epoch(epoch, type='train', attacker=None, optimizer=self.optimizer)
                test_loss, test_accuracy = self.one_epoch(epoch, type='test', attacker=None, optimizer=None)
            # adversarial training
            else:
                # train on normal data
                train_loss, train_accuracy = self.one_epoch(epoch, type='train', attacker=None, optimizer=self.optimizer)
                # train on adversarial data
                adv_train_loss, adv_train_accuracy = self.one_epoch(epoch, type='train', attacker=attacker, optimizer=self.optimizer)
                adv_training_loss_seq.append(adv_train_loss)
                adv_training_accuracy_seq.append(adv_train_accuracy)
                # test on normal data
                test_loss, test_accuracy = self.one_epoch(epoch, type='test', attacker=None, optimizer=None)
                # test on adversarial data
                adv_test_loss, adv_test_accuracy = self.one_epoch(epoch, type='test', attacker=attacker, optimizer=None)
                adv_testing_loss_seq.append(adv_test_loss)
                adv_testing_accuracy_seq.append(adv_test_accuracy)
            training_loss_seq.append(train_loss)
            training_accuracy_seq.append(train_accuracy)
            testing_loss_seq.append(test_loss)
            testing_accuracy_seq.append(test_accuracy)
            self.scheduler.step()
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
                "testing_best_accuracy": testing_best_accuracy,
                "adv_training_accuracy_seq": adv_training_accuracy_seq,
                "adv_training_loss_seq":adv_training_loss_seq,
                "adv_testing_accuracy_seq": adv_testing_accuracy_seq,
                "adv_testing_loss_seq": adv_testing_loss_seq
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
