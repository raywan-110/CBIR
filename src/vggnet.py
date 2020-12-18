# -*- coding: utf-8 -*-

from __future__ import print_function
from lsh import LSHash
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG

from six.moves import cPickle  # 序列化
import csv
import numpy as np
import imageio
import os
import time

from evaluate import evaluate_class
from DB import Database
'''
  downloading problem in mac OSX should refer to this answer:
    https://stackoverflow.com/a/42334357
'''

# configs for histogram
VGG_model = 'vgg19'  # model type
pick_layer = 'avg'  # extract feature of this layer
d_type = 'd1'  # distance type
feat_dim = 512  # 输出特征的维度
# TODO 建议把特征提取的选择也加到一个文件中，方便后面整合做UI
depth = 3  # retrieved depth, set to None will count the ap for whole database 返回top depth张图像
''' MMAP
     depth
      depthNone, vgg19,avg,d1, MMAP 0.688624709114
      depth100,  vgg19,avg,d1, MMAP 0.754443491363
      depth30,   vgg19,avg,d1, MMAP 0.838298388513
      depth10,   vgg19,avg,d1, MMAP 0.913892057193
      depth5,    vgg19,avg,d1, MMAP 0.936158333333
      depth3,    vgg19,avg,d1, MMAP 0.941666666667
      depth1,    vgg19,avg,d1, MMAP 0.934

      (exps below use depth=None)

      vgg19,fc1,d1, MMAP 0.245548035893 (w/o subtract mean)
      vgg19,fc1,d1, MMAP 0.332583126964
      vgg19,fc1,co, MMAP 0.333836506148
      vgg19,fc2,d1, MMAP 0.294452201395
      vgg19,fc2,co, MMAP 0.297209571796
      vgg19,avg,d1, MMAP 0.688624709114
      vgg19,avg,co, MMAP 0.674217021273
'''

use_gpu = torch.cuda.is_available()
means = np.array([103.939, 116.779, 123.68
                  ]) / 255.  # mean of three channels in the order of BGR

# cache dir
cache_dir = 'cache'
lshCache_dir = 'lshCache'
if not os.path.exists(lshCache_dir):
    os.makedirs(lshCache_dir)
# 记录实验结果
result_dir = 'result'
result_csv = 'result.csv'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


class VGGNet(VGG):
    def __init__(self,
                 pretrained=True,
                 model='vgg16',
                 requires_grad=False,
                 remove_fc=False,
                 show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]
        self.fc_ranges = ((0, 2), (2, 5), (5, 7))
        if pretrained:
            exec(
                "self.load_state_dict(models.%s(pretrained=True).state_dict())"
                % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

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
        # fc输出的效果不行
        x = x.view(x.size(0), -1)  # flatten()
        dims = x.size(1)
        if dims >= 25088:
            x = x[:, :25088]
            for idx in range(len(self.fc_ranges)):
                for layer in range(self.fc_ranges[idx][0],
                                   self.fc_ranges[idx][1]):
                    x = self.classifier[layer](x)
                output["fc%d" % (idx + 1)] = x
        else:
            w = self.classifier[0].weight[:, :dims]
            b = self.classifier[0].bias
            x = torch.matmul(x, w.t()) + b
            x = self.classifier[1](x)  # 过fc层
            output["fc1"] = x  # 存储经过fc后的特征
            for idx in range(1, len(self.fc_ranges)):
                for layer in range(self.fc_ranges[idx][0],
                                   self.fc_ranges[idx][1]):
                    x = self.classifier[layer](x)
                output["fc%d" % (idx + 1)] = x

        return output


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


class VGGNetFeat(object):
    def make_samples(self, db, retrival_mode, verbose=True):
        mode = retrival_mode  # 检索模式
        sample_cache = '{}-{}'.format(VGG_model, pick_layer)
        try:
            samples = cPickle.load(
                open(os.path.join(cache_dir, sample_cache), "rb",
                     True))  # 从cache中读入并恢复python对象
            # 重新计算每一个图像的特征(理论上不需要)，并将python对象序列化存入缓存
            for sample in samples:
                sample['hist'] /= np.sum(sample['hist'])  # normalize
            cPickle.dump(samples,
                         open(os.path.join(cache_dir, sample_cache), "wb",
                              True))  # 向cache中存入normalize过后的featrues
            if verbose:
                print(
                    retrival_mode, " mode:",
                    "Using cache..., config=%s, distance=%s, depth=%s" %
                    (sample_cache, d_type, depth))
        # 没有则生成特征描述文件
        except:
            if verbose:
                print(
                    "Counting histogram..., config=%s, distance=%s, depth=%s" %
                    (sample_cache, d_type, depth))

            vgg_model = VGGNet(requires_grad=False, model=VGG_model)
            vgg_model.eval()
            if use_gpu:
                vgg_model = vgg_model.cuda()
            samples = []  # 构造存储特征的列表
            data = db.get_data()
            for d in data.itertuples():  # 组成(Index,img,cls)的元组
                d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
                # 将图像读出来保存在numpy类型数组
                img = imageio.imread(
                    d_img, pilmode="RGB")  # 必须设置pilmode关键字参数为"RGB",否则无法处理灰度图
                img = img[:, :, ::-1]  # switch to BGR 第三个维度倒序
                img = np.transpose(img, (2, 0, 1)) / 255.  # 转置成C*H*W
                img[0] -= means[0]  # reduce B's mean
                img[1] -= means[1]  # reduce G's mean
                img[2] -= means[2]  # reduce R's mean
                img = np.expand_dims(img, axis=0)  # 增加维度，变成1*C*H*W，方便送入VGG
                try:
                    if use_gpu:
                        inputs = torch.autograd.Variable(
                            torch.from_numpy(img).cuda().float())
                    else:
                        inputs = torch.autograd.Variable(
                            torch.from_numpy(img).float())
                    d_hist = vgg_model(inputs)[pick_layer]  # 得到预处理后的图像的输出特征
                    d_hist = np.sum(d_hist.data.cpu().numpy(), axis=0)
                    d_hist /= np.sum(d_hist)  # normalize
                    samples.append({
                        'img': d_img,  # y原图像
                        'cls': d_cls,  # 类别标签
                        'hist': d_hist  # 特征
                    })
                except:
                    pass
            cPickle.dump(samples,
                         open(os.path.join(cache_dir, sample_cache), "wb",
                              True))  # 序列化后存入缓存中
        # 选择检索模式
        if mode == 'LSH':
            try:
                lsh = cPickle.load(
                    open(os.path.join(lshCache_dir, sample_cache), "rb",
                         True))  # 读入hashtable
            except:
                # 需要重新生成
                lsh = LSHash(hash_size=12,input_dim=feat_dim,num_hashtables=3)
                for i, sample in enumerate(samples):
                    input_vec = sample['hist']
                    # extra = {'img': sample['img'], 'cls': sample['cls']}
                    extra = (sample['img'], sample['cls'])
                    lsh.index(input_vec.flatten(), extra_data=extra)  # 哈希表中存储结构：[((vec),img,cls)]
                cPickle.dump(lsh,
                             open(os.path.join(lshCache_dir, sample_cache),
                                  "wb", True))  # 序列化后存入缓存中
            return samples, lsh
        else:
            return samples


if __name__ == "__main__":
    # evaluate database
    db = Database()
    start = time.time()
    APs = evaluate_class(db, f_class=VGGNetFeat, d_type=d_type,
                         depth=depth)  # 检索top-3图片,返回平均准确率
    end = time.time()

    cls_MAPs = []
    with open(os.path.join(result_dir, result_csv), 'w', encoding='UTF-8') as f:
        f.write("Vgg-LSH-cosine reusult: MAP&MMAP")
        for cls, cls_APs in APs.items():
            MAP = np.mean(cls_APs)
            print("Class {}, MAP {}".format(cls, MAP))
            f.write("\nClass {}, MAP {}".format(cls, MAP))
            cls_MAPs.append(MAP)
        print("MMAP", np.mean(cls_MAPs))
        f.write("\nMMAP {}".format(np.mean(cls_MAPs)))
        print("total time:",end - start)
        f.write("\ntotal time:{0:.4f}s".format(end - start))
