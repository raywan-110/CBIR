from __future__ import print_function

# from network import VGGNet
from dirtorch.utils import common
import dirtorch.nets as nets

import pandas as pd
import faiss
import torch
import torchvision.transforms as transforms
import torch.nn as nn

from six.moves import cPickle
import numpy as np
import imageio
import os
import time
from PIL import Image

from evaluate import evaluate_class
from DB import Database


def load_model(path, iscuda):
    checkpoint = common.load_checkpoint(path, iscuda)
    net = nets.create_model(pretrained="", **checkpoint['model_options'])
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    # if 'pca' in checkpoint:
    #     net.pca = checkpoint.get('pca')
    return net


use_gpu = torch.cuda.is_available()
# torch.cuda.set_device(2)

# cache dir
cache_dir = '../cache'

result_dir = 'result'
result_csv = 'vgg16.csv'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

dic_addr = 'res101_AP_GeM-oxf-dict'
vec_addr = 'res101_AP_GeM-oxf-vec'
index_addr = 'res101_AP_GeM-oxf-index'
# dic_addr = 'vgg-oxf-dict'
# vec_addr = 'vgg-oxf-vec'
# index_addr = 'vgg-oxf-index'
depth = 10

# LOAD_MODEL_PATH = None
# LOAD_MODEL_PATH = '../model/imagenet-caffe-vgg16-features-d369c8e.pth'
# LOAD_MODEL_PATH = '../model/imagenet-caffe-resnet101-features-10a101d.pth'
# LOAD_WHITEN_PATH = '../model/retrieval-SfM-120k-resnet101-gem-whiten-22ab0c1.pth'
CHECKPOINT = "../model/Resnet-101-AP-GeM.pt"

IMAGE_NORMALIZER = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

REMOVE_FC = False


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
                print("Using cache..., config=%s, depth=%s" % (vec_addr, depth))

        except:
            if verbose:
                print("Counting histogram..., config=%s, depth=%s" % (vec_addr, depth))

            # base_model = VGGNet(load_features_path=LOAD_MODEL_PATH, requires_grad=False)
            # base_model = Res101(load_features_path=LOAD_MODEL_PATH,
            #                     use_Gem_whiten=True, load_whiten_path=LOAD_WHITEN_PATH)
            # base_model =
            base_model = load_model(CHECKPOINT, False)
            base_model.eval()
            print("load successfully!")
            if REMOVE_FC:
                base_model = nn.Sequential(*list(base_model.children())[:-1])
                print("Remove FC")

            if use_gpu:
                base_model = base_model.cuda()

            vecbase = []
            dicbase = []
            data = db.get_data()
            count = 1
            for d in data.itertuples():
                # if count == 5:
                #     break
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

                d_hist = base_model(inputs).view(-1, )
                d_hist = d_hist.data.cpu().numpy()

                vecbase.append(d_hist)
                dicbase.append((d_cls, d_img))

                print(count)
                count += 1

            vecbase = np.array(vecbase).astype('float32')
            print(vecbase.shape)
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
    APs = evaluate_class(db, f_class=ModelFeat, depth=depth)
    end = time.time()

    cls_MAPs = []
    with open(os.path.join(result_dir, result_csv), 'w', encoding='UTF-8') as f:
        f.write("Vgg16-oxf-cosine result: MAP&MMAP")
        for cls, cls_APs in APs.items():
            MAP = np.mean(cls_APs)
            print("Class {}, MAP {}".format(cls, MAP))
            f.write("\nClass {}, MAP {}".format(cls, MAP))
            cls_MAPs.append(MAP)
        print("MMAP", np.mean(cls_MAPs))
        f.write("\nMMAP {}".format(np.mean(cls_MAPs)))
        print("total time:", end - start)
        f.write("\ntotal time:{0:.4f}s".format(end - start))
