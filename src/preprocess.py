import numpy as np
import imageio
import os
from DB import Database
from PIL import Image

db = Database()
data = db.get_data()

count = 1
img_l = []
for d in data.itertuples():
    # if count == 5:
    #     break
    d_img, d_cls = getattr(d, "img"), getattr(d, "cls")

    new = None

    img = imageio.imread(d_img, pilmode="RGB")
    img = Image.fromarray(img)
    img = np.array(img)
    h, w = img.shape[0], img.shape[1]
    if h * w > 1048576:
        new = img[:min(1024, h), :min(1024, w), :]

    if new is not None:
        imageio.imwrite(d_img, new)
        print(d_img)
        img_l.append(d_img)
    print(count)
    count += 1

with open(os.path.join("caijian.txt"), 'w', encoding='UTF-8') as file:
    for e in img_l:
        file.write("{}\n".format(e))