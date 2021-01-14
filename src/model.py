from __future__ import print_function

from network import VGGNet
import pandas as pd
import faiss
import torch
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

from six.moves import cPickle
import numpy as np
import imageio
import os
import time
from PIL import Image

from evaluate import evaluate_class
from DB import Database

use_gpu = torch.cuda.is_available()

# cache dir
cache_dir = '../cache'

result_dir = 'result'
result_csv = 'vgg19.csv'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

dic_addr = 'vgg19-oxf-dict'
vec_addr = 'vgg19-oxf-vec'
index_addr = 'vgg19-oxf-index'
depth = 10
d_type = 'cosine'  # distance type

feat_dim = 512
# LOAD_MODEL_PATH = 'trained_model/model_m0_5.pth'
LOAD_MODEL_PATH = None
pick_layer = 'avg'

IMAGE_NORMALIZER = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class ModelFeat(object):
    @staticmethod
    def make_samples(db, mode, verbose=True):
        try:
            dicbase = cPickle.load(open(os.path.join(cache_dir, dic_addr), "rb", True))
            vecbase = cPickle.load(open(os.path.join(cache_dir, vec_addr), "rb", True))
            if mode == 'Linear':
                index = faiss.read_index(os.path.join(cache_dir, index_addr))
            else:
                raise ValueError("you should choose a correct retrival mode")
            if verbose:
                print("Using cache..., config=%s, distance=%s, depth=%s" % (vec_addr, d_type, depth))

        except:
            if verbose:
                print("Counting histogram..., config=%s, distance=%s, depth=%s" % (vec_addr, d_type, depth))

            base_model = VGGNet(load_model_path=LOAD_MODEL_PATH, requires_grad=False)
            if use_gpu:
                base_model = base_model.cuda()
            vecbase = []
            dicbase = []
            data = db.get_data()
            for d in data.itertuples():
                d_img, d_cls = getattr(d, "img"), getattr(d, "cls")

                img = imageio.imread(d_img, pilmode="RGB")

                img = Image.fromarray(img)
                img = IMAGE_NORMALIZER(img)
                img = np.array(img)

                img = np.expand_dims(img, axis=0)

                if use_gpu:
                    inputs = torch.autograd.Variable(torch.from_numpy(img).cuda().float())
                else:
                    inputs = torch.autograd.Variable(torch.from_numpy(img).float())
                d_hist = base_model(inputs)[pick_layer]

                d_hist = np.sum(d_hist.data.cpu().numpy(), axis=0)
                d_hist /= np.sum(d_hist) + 1e-15  # normalize
                vecbase.append(d_hist)
                dicbase.append((d_cls, d_img))

            vecbase = np.array(vecbase).astype('float32')
            d = vecbase.shape[1]
            dicbase = pd.DataFrame(dicbase, columns=['cls', 'img'])
            if mode == 'Linear':
                index = faiss.IndexFlatL2(d)
                index.add(vecbase)
            else:
                raise ValueError("you should choose a correct retrival mode")
            cPickle.dump(dicbase, open(os.path.join(cache_dir, dic_addr), "wb", True))
            cPickle.dump(vecbase, open(os.path.join(cache_dir, vec_addr), "wb", True))
            faiss.write_index(index, os.path.join(cache_dir, index_addr))

        return index, dicbase, vecbase


if __name__ == "__main__":
    # evaluate database
    db = Database()
    start = time.time()
    APs = evaluate_class(db, f_class=ModelFeat, d_type=d_type, depth=depth)
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
