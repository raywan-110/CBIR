# -*- coding: utf-8 -*-

from __future__ import print_function

# from scipy import spatial
import numpy as np
from scipy import spatial
mode = 'Linear'


class Evaluation(object):
    def make_samples(self):
        raise NotImplementedError("Needs to implemented this method")


def distance(v1, v2, d_type='d1'):
    assert v1.shape == v2.shape, "shape of two vectors need to be same!"
    # L1范数 汉明距离
    if d_type == 'd1':
        return np.sum(np.absolute(v1 - v2))
    # L2范数 欧氏距离
    elif d_type == 'd2':
        return np.sum((v1 - v2) ** 2)
    # 正则化的欧氏距离 
    elif d_type == 'd2-norm':
        return 2 - 2 * np.dot(v1, v2)
    elif d_type == 'd3':
        pass
    elif d_type == 'd4':
        pass
    elif d_type == 'd5':
        pass
    elif d_type == 'd6':
        pass
    elif d_type == 'd7':
        return 2 - 2 * np.dot(v1, v2)
    elif d_type == 'd8':
        return 2 - 2 * np.dot(v1, v2)
    elif d_type == 'cosine':
        return spatial.distance.cosine(v1, v2)
    elif d_type == 'square':
        return np.sum((v1 - v2) ** 2)


def AP(label, results, dicbase):
    ''' infer a query, return it's ap

    arguments
      label  : query's class
      results: a list of ID, e.g.:[1,2,11,4]
    '''
    precision = []
    hit = 0  # 命中的个数
    for i, result in enumerate(results):
        if dicbase.iloc[result,0] == label:
            hit += 1
            precision.append(hit / (i + 1.))
    if hit == 0:
        return 0.
    return np.mean(precision)


# 测试检索性能
def infer(query,
          mode,
          samples=None,
          lsh=None,
          sample_db_fn=None,
          depth=None,
          d_type='d1'):
    ''' infer a query, return it's ap

    arguments
      query       : a dict with three keys, see the template
                    {
                      'img': <path_to_img>,
                      'cls': <img class>,
                      'hist' <img histogram>
                    }
      samples     : a list of {
                                'img': <path_to_img>,
                                'cls': <img class>,
                                'hist' <img histogram>
                              }
      sample_db_fn: a function making samples, should be given if Database != None
      depth       : retrieved depth during inference, the default depth is equal to database size
      d_type      : distance type
  '''
    # TODO 选择检索模式
    q_img, q_cls, q_hist = query['img'], query['cls'], query['hist']
    if mode == 'LSH':
        results = lsh.query(query_point=q_hist, img_addr=q_img, num_results=depth, distance_func="cosine")
    else:
        results = []
        dis_l = []
        for idx, sample in enumerate(samples):
            s_img, s_cls, s_hist = sample['img'], sample['cls'], sample['hist']
            if q_img == s_img:  # 相同的图片不算
                continue
            if idx < depth:
                results.append({
                    'img': s_img,
                    'dis': distance(q_hist, s_hist, d_type=d_type),
                    'cls': s_cls
                })
                dis_l.append(results[-1]['dis'])
                max_dis = max(dis_l)
                max_index = dis_l.index(max_dis)
            else:
                dis = distance(q_hist, s_hist, d_type=d_type)
                if dis < max_dis:
                    results[max_index] = {
                        'img': s_img,
                        'dis': dis,
                        'cls': s_cls
                    }
                    dis_l[max_index] = dis
                    max_dis = max(dis_l)
                    max_index = dis_l.index(max_dis)

        results = sorted(results, key=lambda x: x['dis'])
        if depth and depth < len(results):
            results = results[:depth]  # 返回top-k检索结果
    #####
    ap = AP(q_cls, results, sort=False)

    return ap, results


def evaluate(db, sample_db_fn, depth=None, d_type='d1'):
    ''' infer the whole database

    arguments
      db          : an instance of class Database
      sample_db_fn: a function making samples, should be given if Database != None
      depth       : retrieved depth during inference, the default depth is equal to database size
      d_type      : distance type
  '''
    classes = db.get_class()
    print(classes)
    ret = {c: [] for c in classes}

    samples = sample_db_fn(db)  # 提前生成好samples

    for query in samples:
        ap, _ = infer(query, samples=samples, depth=depth, d_type=d_type)
        ret[query['cls']].append(ap)

    return ret


def evaluate_class(db, f_class=None, f_instance=None, depth=None):
    ''' infer the whole database

    arguments
      db     : an instance of class Database
      f_class: a class that generate features, needs to implement make_samples method
      depth  : retrieved depth during inference, the default depth is equal to database size
      d_type : distance type
  '''
    assert f_class or f_instance, "needs to give class_name or an instance of class"

    classes = db.get_class()
    ret = {c: [] for c in classes}

    if f_class:
        f = f_class()
    elif f_instance:
        f = f_instance
    index, dicbase, vecbase = f.make_samples(db, mode)  # 调用f的make_samples的方法
    query = vecbase
    query_label = dicbase.iloc[:,0]  # 标签
    result_D, result_I = index.search(query, depth + 1)     # actual search
    # 去除自身
    result_D = result_D[:,1:]  
    result_I = result_I[:,1:]
    for results, label in zip(result_I, query_label):
        ap = AP(label,results,dicbase)
        ret[label].append(ap)  # 记录该类别的ap
    return ret
