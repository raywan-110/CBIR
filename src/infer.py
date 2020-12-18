# -*- coding: utf-8 -*-

from __future__ import print_function
import time

from evaluate import infer
from DB import Database

from vggnet import VGGNetFeat
from resnet import ResNetFeat

depth = 3
d_type = 'd1'
query_idx = 100
mode1 = 'LSH'
# 测试infer函数
if __name__ == '__main__':
    db = Database()

    # retrieve by VGG
    method = VGGNetFeat()
    samples, lsh = method.make_samples(db, mode1)
    query = samples[query_idx]
    # 计算检索时间
    start = time.time()
    _, result = infer(query, mode1, samples=lsh, depth=depth, d_type=d_type)
    end = time.time()
    print("vgg consuming time:{0:.3f}s".format(end - start))
    print("query class:",query['cls'])
    print(result)
    # retrieve by resnet
    # method = ResNetFeat()
    # samples = method.make_samples(db)
    # query = samples[query_idx]
    # start = time.time()
    # _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
    # end = time.time()
    # print("resnet consuming time:", end - start)
    # print(result)

