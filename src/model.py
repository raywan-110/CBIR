from __future__ import print_function

from triplet_net import VGGNet, NetVLAD, EmbedNet

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

use_gpu = torch.cuda.is_available()
means = np.array([103.939, 116.779, 123.68]) / 255.  # mean of three channels in the order of BGR

# cache dir
cache_dir = 'cache'
lsh_Cache_dir = 'lshCache'
if not os.path.exists(lsh_Cache_dir):
    os.makedirs(lsh_Cache_dir)
# 记录实验结果
result_dir = 'result'
result_csv = 'vgg19_finetune_lsh_simple.csv'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

model = 'vgg19_f_lsh_onsim'
depth = 10
d_type = 'cosine'  # distance type
feat_dim = 512  # 输出特征的维度
VGG_model = 'vgg19'
LOAD_MODEL_PATH = 'trained_model/model_simple.pth'
# LOAD_MODEL_PATH = None


class ModelFeat(object):
    def make_samples(self, db, verbose=True):
        sample_cache = model
        try:
            samples = cPickle.load(open(os.path.join(cache_dir, sample_cache), "rb", True))  # 从cache中读入并恢复python对象
            if verbose:
                print("Using cache..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
        # 没有则生成特征描述文件
        except:
            if verbose:
                print("Counting histogram..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))

            base_model = VGGNet(load_model_path=LOAD_MODEL_PATH, model=VGG_model, requires_grad=False)
            # dim = list(base_model.parameters())[-1].shape[0]
            # embed_model = EmbedNet(base_model, NetVLAD(num_clusters=32, dim=dim, alpha=1.0))
            # embed_model.eval()
            if use_gpu:
                # embed_model = embed_model.cuda()
                base_model = base_model.cuda()
            samples = []  # 构造存储特征的列表
            data = db.get_data()
            for d in data.itertuples():  # 组成(Index,img,cls)的元组
                d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
                # 将图像读出来保存在numpy类型数组
                img = imageio.imread(d_img, pilmode="RGB")  # 必须设置pilmode关键字参数为"RGB",否则无法处理灰度图
                img = img[:, :, ::-1]  # switch to BGR 第三个维度倒序
                img = np.transpose(img, (2, 0, 1)) / 255.  # 转置成C*H*W
                img[0] -= means[0]  # reduce B's mean
                img[1] -= means[1]  # reduce G's mean
                img[2] -= means[2]  # reduce R's mean
                img = np.expand_dims(img, axis=0)  # 增加维度，变成1*C*H*W，方便送入VGG
                try:
                    if use_gpu:
                        inputs = torch.autograd.Variable(torch.from_numpy(img).cuda().float())
                    else:
                        inputs = torch.autograd.Variable(torch.from_numpy(img).float())
                    d_hist = base_model(inputs)['avg']  # 得到预处理后的图像的输出特征
                    d_hist = np.sum(d_hist.data.cpu().numpy(), axis=0)
                    d_hist /= np.sum(d_hist) + 1e-15  # normalize
                    samples.append({
                        'img': d_img,  # y原图像
                        'cls': d_cls,  # 类别标签
                        'hist': d_hist  # 特征
                    })
                except:
                    pass
            cPickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb", True))  # 序列化后存入缓存中

        try:
            lsh = cPickle.load(open(os.path.join(lsh_Cache_dir, sample_cache), "rb", True))  # 读入hashtable
        except:
            # 需要重新生成
            lsh = LSHash(hash_size=8, input_dim=feat_dim, num_hashtables=4)
            for i, sample in enumerate(samples):
                input_vec = sample['hist']
                extra = (sample['img'], sample['cls'])
                lsh.index(input_vec.flatten(), extra_data=extra)  # 哈希表中存储结构：[((vec),img,cls)]
            cPickle.dump(lsh, open(os.path.join(lsh_Cache_dir, sample_cache), "wb", True))  # 序列化后存入缓存中
        return samples, lsh


if __name__ == "__main__":
    # evaluate database
    db = Database()
    start = time.time()
    APs = evaluate_class(db, f_class=ModelFeat, d_type=d_type, depth=depth)  # 检索top-10图片,返回平均准确率
    end = time.time()

    cls_MAPs = []
    with open(os.path.join(result_dir, result_csv), 'w', encoding='UTF-8') as f:
        f.write("Vgg19-simple-cosine result: MAP&MMAP")
        for cls, cls_APs in APs.items():
            MAP = np.mean(cls_APs)
            print("Class {}, MAP {}".format(cls, MAP))
            f.write("\nClass {}, MAP {}".format(cls, MAP))
            cls_MAPs.append(MAP)
        print("MMAP", np.mean(cls_MAPs))
        f.write("\nMMAP {}".format(np.mean(cls_MAPs)))
        print("total time:", end - start)
        f.write("\ntotal time:{0:.4f}s".format(end - start))
