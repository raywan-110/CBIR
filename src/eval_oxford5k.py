from retrieval import search
import os
from DB import Database
from network import VGGNet
from model import ModelFeat

mode1 = 'Linear'
LOAD_MODEL_PATH = None
gt_dir = "../eval_oxford5k/gt_files_170407"
oxford5k_dir = "../oxbuild_images"
ranked_list_dir = "../eval_oxford5k/ranked_list"

db = Database()
method = ModelFeat
index, dicbase, _ = method.make_samples(db=db, mode="Linear")

model = VGGNet(load_model_path=LOAD_MODEL_PATH, requires_grad=False)
model.eval()

for f in os.listdir(gt_dir):
    if not f.endswith("query.txt"):
        continue

    query_id = f[:-10]  # 序号
    img = open(os.path.join(gt_dir, f)).readlines()[0].split(" ")[0][5:] + ".jpg"
    img_path = os.path.join(oxford5k_dir, img)
    results = search(model, img_path, index, dicbase)
    
    ranked_list = []
    for e in results:
        ranked_list.append(e.split('/')[-1][:-4])

    with open(os.path.join(ranked_list_dir, query_id+".txt"), 'w', encoding='UTF-8') as file:
        for img_name in ranked_list:
            file.write("{}\n".format(img_name))
