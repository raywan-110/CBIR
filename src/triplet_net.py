# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.vgg import VGG
import time
import csv
import numpy as np
import torch.optim as optim
import imageio
import os
import time
# from itertools import ifilter
from hard_triplet_loss import HardTripletLoss

'''
  downloading problem in mac OSX should refer to this answer:
    https://stackoverflow.com/a/42334357
'''

# configs for histogram
VGG_model = 'vgg19'  # model type
pick_layer = 'avg'  # extract feature of this layer
d_type = 'cosine'  # distance type

means = np.array([103.939, 116.779, 123.68]) / 255.  # mean of three channels in the order of BGR


class VGGNet(VGG):
    def __init__(self, load_model_path=None, pretrained=True, model='vgg19',
                 requires_grad=False, remove_fc=False, show_params=False):
        super().__init__(make_layers(cfg[model]))
        # self.ranges = ranges[model]  # 貌似没有用
        # self.fc_ranges = ((0, 2), (2, 5), (5, 7))
        self.use_gpu = torch.cuda.is_available()
        if load_model_path is not None:
            del self.avgpool
            del self.classifier
            self.load_state_dict(torch.load(load_model_path))
            print("Load model successfully!")
        elif pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)
            del self.avgpool
            del self.classifier

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False
        #
        # if remove_fc:  # delete redundant fully-connected layer params, can save memory
        #     del self.classifier
        if self.use_gpu:
            self.cuda()

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        x = self.features(x)
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


# 设置 vgg的参数
ranges = {
    'vgg11': ((0, 3), (3, 6), (6, 11), (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13':
        [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'vgg19': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
        512, 'M', 512, 512, 512, 512, 'M'
    ],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=512, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


class EmbedNet(nn.Module):
    def __init__(self, base_model, net_vlad):
        super(EmbedNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad

    def forward(self, x):
        x = self.base_model(x)
        embedded_x = self.net_vlad(x)
        return embedded_x


class TripletNet(nn.Module):
    def __init__(self, embed_net):
        super(TripletNet, self).__init__()
        self.embed_net = embed_net

    def forward(self, a, p, n):
        embedded_a = self.embed_net(a)
        embedded_p = self.embed_net(p)
        embedded_n = self.embed_net(n)
        return embedded_a, embedded_p, embedded_n

    def feature_extract(self, x):
        return self.embed_net(x)


# test
if __name__ == '__main__':
    # base_model = VGGNet()
    # netvlad = NetVLAD()
    # model = EmbedNet(base_model, netvlad)
    # # tripletnet = TripletNet(model)
    #
    # # img1 = torch.zeros((1, 3, 100, 100))
    # # img2 = torch.zeros((1, 3, 100, 100))
    # # img3 = torch.zeros((1, 3, 100, 100))
    # # res = tripletnet(img1)
    # #
    # # print(res.shape)
    #
    # dim = list(base_model.parameters())[-1].shape[0]  # last channels (512)

    # Define model for embedding
    # net_vlad = NetVLAD(num_clusters=32, dim=dim, alpha=1.0)
    torch.cuda.set_device(1)
    model = VGGNet()
    # # Define loss
    criterion = HardTripletLoss(margin=0.1).cuda()
    #
    # # This is just toy example. Typically, the number of samples in each classes are 4.
    labels = torch.randint(0, 10, (40,)).long()
    x = torch.rand(40, 3, 128, 128).cuda()
    output = model(x)
    print(output['avg'].shape)
    triplet_loss = criterion(output['avg'], labels.cuda())
    print(triplet_loss)
    print(triplet_loss.data)
