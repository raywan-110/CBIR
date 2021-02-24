import sys
import os
import numpy as np
from typing import List


def load_list(fname: str):
    """Plain text list loader. Reads from file separated by newlines, and returns a
    list of the file with whitespaces stripped.
    Args:
        fname (str): Name of file to be read.
    Returns:
        List[str]: A stripped list of strings, using newlines as a seperator from file.
    """

    return [e.strip() for e in open(fname, 'r').readlines()]


def compute_ap(pos: List[str], amb: List[str], ranked_list: List[str]):
    """Compute average precision against a retrieved list of images. There are some bits that
    could be improved in this, but is a line-to-line port of the original C++ benchmark code.
    Args:
        pos (List[str]): List of positive samples. This is normally a conjugation of
        the good and ok samples in the ground truth data.
        amb (List[str]): List of junk samples. This is normally the junk samples in
        the ground truth data. Omitting this makes no difference in the AP.
        ranked_list (List[str]): List of retrieved images from query to be evaluated.
    Returns:
        float: Average precision against ground truth - range from 0.0 (worst) to 1.0 (best).
    """

    intersect_size, old_recall, ap = 0.0, 0.0, 0.0
    old_precision, j = 1.0, 1.0

    for e in ranked_list:
        if e in amb:
            continue

        if e in pos:
            intersect_size += 1.0

        recall = intersect_size / len(pos)
        precision = intersect_size / j
        ap += (recall - old_recall) * ((old_precision + precision) / 2.0)

        old_recall = recall
        old_precision = precision
        j += 1.0

    return ap


ranked_list_dir = "ranked_list"
gt_dir = "gt_files_170407"

#  ranked list统一保存成：all_souls_5_rankedlist.txt这种形式
if __name__ == '__main__':
    ap_l = []
    with open('AP_list.txt', 'w', encoding='UTF-8') as f:
        for m in sorted(os.listdir(ranked_list_dir)):
            ranked_list = load_list(os.path.join(ranked_list_dir, m))
            gt_query = m[:-4]  # 去掉结尾的"rankedlist.txt"
            print(gt_query)

            pos_set = list(set(load_list(os.path.join(gt_dir, "%s_good.txt" % gt_query)) +
                               load_list(os.path.join(gt_dir, "%s_ok.txt" % gt_query))))
            junk_set = load_list(os.path.join(gt_dir, "%s_junk.txt" % gt_query))
            ap = compute_ap(pos_set, junk_set, ranked_list)
            ap_l.append(ap)
            print(gt_query, "\t", ap)
            f.write("{}\t{}\n".format(gt_query, ap))

        mAP = np.mean(ap_l)
        print("mAP\t{}".format(mAP))
        f.write("mAP\t{}".format(mAP))
