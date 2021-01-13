from __future__ import print_function

from network import VGGNet
import pandas as pd
import faiss
import torch
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

from six.moves import cPickle  # 序列化
import numpy as np
import imageio
import os
import time
from PIL import Image

from evaluate import evaluate_class
from DB import Database

use_gpu = torch.cuda.is_available()

# cache dir
cache_dir = 'cache'
lsh_Cache_dir = 'lshCache'
if not os.path.exists(lsh_Cache_dir):
    os.makedirs(lsh_Cache_dir)
# 记录实验结果
result_dir = 'result'
result_csv = 'vgg19.csv'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
# 向量数据库和字典数据库(存放标签和图片地址)的地址
dic_addr = 'vgg19-sim-dict'
vec_addr = 'vgg19-sim-vec'
depth = 10
d_type = 'cosine'  # distance type

feat_dim = 512  # 输出特征的维度
# LOAD_MODEL_PATH = 'trained_model/model_m0_5.pth'
LOAD_MODEL_PATH = None
pick_layer = 'avg'

IMAGE_NORMALIZER = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class ModelFeat(object):
    def make_samples(self, db, mode, verbose=True):
        try:
            dicbase =cPickle.load(open(os.path.join(cache_dir, dic_addr), "rb", True))
            vecbase =cPickle.load(open(os.path.join(cache_dir, vec_addr), "rb", True))
            if mode == 'Linear':
                index_addr = 'vgg19-sim-index' 
                index = faiss.read_index(os.path.join(cache_dir, index_addr))
            else:
                raise ValueError("you should choose a correct retrival mode")
            if verbose:
                print("Using cache..., config=%s, distance=%s, depth=%s" % (vec_addr, d_type, depth))
        # 没有则生成特征描述文件
        except:
            if verbose:
                print("Counting histogram..., config=%s, distance=%s, depth=%s" % (vec_addr, d_type, depth))

            base_model = VGGNet(load_model_path=LOAD_MODEL_PATH, requires_grad=False)
            if use_gpu:
                base_model = base_model.cuda()
            vecbase = []
            dicbase = []
            data = db.get_data()
            for d in data.itertuples():  # 组成(Index,img,cls)的元组
                d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
                # 将图像读出来保存在numpy类型数组
                img = imageio.imread(d_img, pilmode="RGB")  # 必须设置pilmode关键字参数为"RGB",否则无法处理灰度图

                img = Image.fromarray(img)
                img = IMAGE_NORMALIZER(img)
                img = np.array(img)

                img = np.expand_dims(img, axis=0)  # 增加维度，变成1*C*H*W，方便送入VGG
                try:
                    if use_gpu:
                        inputs = torch.autograd.Variable(torch.from_numpy(img).cuda().float())
                    else:
                        inputs = torch.autograd.Variable(torch.from_numpy(img).float())
                    d_hist = base_model(inputs)[pick_layer]  # 得到预处理后的图像的输出特征

                    d_hist = np.sum(d_hist.data.cpu().numpy(), axis=0)
                    d_hist /= np.sum(d_hist) + 1e-15  # normalize
                    vecbase.append(d_hist)  # 构建向量数据库
                    dicbase.append((d_cls, d_img))  # 构建字典数据库
                except:
                    pass
            vecbase = np.array(vecbase).astype('float32')  # 转为float32类型的array，用于生成index
            d = vecbase.shape[1]  # 向量的维度
            dicbase = pd.DataFrame(dicbase, columns=['cls','img'])
            if mode == 'Linear':
                index_addr = 'vgg19-sim-index' 
                index = faiss.IndexFlatL2(d)  # 基于L2距离的暴力搜索
                index.add(vecbase)
            else:
                raise ValueError("you should choose a correct retrival mode")
            cPickle.dump(dicbase, open(os.path.join(cache_dir, dic_addr), "wb", True))  # 序列化后存入缓存中
            cPickle.dump(vecbase, open(os.path.join(cache_dir, vec_addr), "wb", True))  # 序列化后存入缓存中
            faiss.write_index(index, os.path.join(cache_dir, index_addr))  # 序列化后存入缓存中

        return index, dicbase, vecbase


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
