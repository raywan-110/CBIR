from retrieval import search
import os
import torch.nn as nn
import time

from DB import Database
from model import ModelFeat
from dirtorch.utils import common
import dirtorch.nets as nets


def load_model(path, iscuda):
    checkpoint = common.load_checkpoint(path, iscuda)
    net = nets.create_model(pretrained="", **checkpoint['model_options'])
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    # if 'pca' in checkpoint:
    #     net.pca = checkpoint.get('pca')
    return net


mode1 = 'Linear'
REMOVE_FC = False


gt_dir = "../eval_oxford5k/gt_files_170407"
oxford5k_dir = "../oxbuild_images"
ranked_list_dir = "../eval_oxford5k/ranked_list"

db = Database()
method = ModelFeat
index, dicbase, _ = method.make_samples(db=db, mode="Linear")

checkpoint = "../model/Resnet-101-AP-GeM.pt"

model = load_model(checkpoint, False)
model.eval()

if REMOVE_FC:
    model = nn.Sequential(*list(model.children())[:-1])
    print("Remove FC")

t = time.time()
for f in os.listdir(gt_dir):
    if not f.endswith("query.txt"):
        continue

    query_id = f[:-10]  # 序号
    img = open(os.path.join(gt_dir, f)).readlines()[0].split(" ")[0][5:] + ".jpg"
    img_path = os.path.join(oxford5k_dir, img)
    results = search(model, img_path, index, dicbase, depth=5063)

    ranked_list = []
    for e in results:
        ranked_list.append(e.split('/')[-1][:-4])

    with open(os.path.join(ranked_list_dir, query_id+".txt"), 'w', encoding='UTF-8') as file:
        for img_name in ranked_list:
            file.write("{}\n".format(img_name))

print("{}s per image".format((time.time() - t)/55))
