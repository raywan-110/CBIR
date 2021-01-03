# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn

from torchvision import models
import numpy as np
import torch.optim as optim
from hard_triplet_loss import HardTripletLoss

'''
  downloading problem in mac OSX should refer to this answer:
    https://stackoverflow.com/a/42334357
'''


class VGGNet(nn.Module):
    def __init__(self, load_model_path=None, requires_grad=True):
        super(VGGNet, self).__init__()
        self.use_gpu = torch.cuda.is_available()
        self.model = models.vgg19(pretrained=True).features

        if load_model_path is not None:
            self.load_state_dict(torch.load(load_model_path))
            print("Load model successfully!")

        if self.use_gpu:
            self.cuda()

        if requires_grad:
            for param in self.parameters():
                param.requires_grad = True
        else:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        output = {}
        x = self.model(x)
        avg_pool = torch.nn.AvgPool2d((x.size(-2), x.size(-1)),
                                      stride=(x.size(-2), x.size(-1)),
                                      padding=0,
                                      ceil_mode=False,
                                      count_include_pad=True)
        avg = avg_pool(x)  # avg.size = N * 512 * 1 * 1
        avg = avg.view(avg.size(0), -1)  # avg.size = N * 512
        output['avg'] = avg  # 存储embedding avg后的特征
        return output

    def fit(self, criterion, learning_rate, weight_decay, train_loader, n_epochs, model_save_path):
        for param in self.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_val = np.inf
        for epoch in range(n_epochs):
            print("*************")
            print("Training Epoch {}".format(epoch))
            print("*************")
            loss_l = []
            train_iter = iter(train_loader)
            n_minibatches = len(train_loader)
            for mini_batch_index in range(n_minibatches):
                images, labels = train_iter.next()
                if self.use_gpu:
                    images = images.cuda()
                    labels = labels.cuda()
                    self.cuda()
                feature = self.forward(images)['avg']
                triplet_loss = criterion(feature, labels)
                loss_l.append(triplet_loss.data)

                optimizer.zero_grad()
                triplet_loss.backward()
                optimizer.step()

                if mini_batch_index % 20 == 0:
                    print('IN EPOCH {} Iter {}: TripletLoss {}'.format(epoch, mini_batch_index, triplet_loss.data))

            mean_loss = torch.tensor(loss_l).sum() / len(loss_l)
            print("best_val: {}, mean_triplet_loss: {}".format(best_val, mean_loss))
            if mean_loss < best_val:
                best_val = mean_loss
                torch.save(self.state_dict(), model_save_path)
                print("save model successfully!")
            print("Epoch {} Mean Loss: {}".format(epoch, mean_loss))


if __name__ == '__main__':
    torch.cuda.set_device(1)
    model = VGGNet()
    # # Define loss
    criterion = HardTripletLoss(margin=0.1).cuda()
    # # This is just toy example. Typically, the number of samples in each classes are 4.
    labels = torch.randint(0, 10, (40,)).long()
    x = torch.rand(40, 3, 128, 128).cuda()
    output = model(x)
    print(output['avg'].shape)
    triplet_loss = criterion(output['avg'], labels.cuda())
    print(triplet_loss)
    print(triplet_loss.data)
