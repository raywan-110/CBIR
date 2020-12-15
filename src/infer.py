# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import infer
from DB import Database

from vggnet import VGGNetFeat
from resnet import ResNetFeat

depth = 5
d_type = 'd1'
query_idx = 0
# 测试infer函数
if __name__ == '__main__':
    db = Database()

    # retrieve by VGG
    method = VGGNetFeat()
    samples = method.make_samples(db)
    query = samples[query_idx]
    _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
    print(result)

    # retrieve by resnet
    method = ResNetFeat()
    samples = method.make_samples(db)
    query = samples[query_idx]
    _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
    print(result)
