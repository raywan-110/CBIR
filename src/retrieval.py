from evaluate import infer
from DB import Database
import numpy as np
from network import VGGNet
from model import ModelFeat
import imageio
import torch
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *
import torchvision.transforms as transforms
from PIL import Image


depth = 10
d_type = 'cosine'
mode1 = 'Linear'
pick_layer = 'avg'  # extract feature of this layer
feat_dim = 512  # 输出特征的维度
# LOAD_MODEL_PATH = 'trained_model/model_simple.pth'
LOAD_MODEL_PATH = None

IMAGE_NORMALIZER = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


if __name__ == '__main__':
    db = Database()
    # img = filedialog.askopenfilename()
    img = 'oxbuild_images/all_souls_000013.jpg'
    # retrieve by VGG
    method = ModelFeat
    samples, lsh = method.make_samples(db, mode1)

    model = VGGNet(load_model_path=LOAD_MODEL_PATH, requires_grad=False)
    model.eval()

    imag = imageio.imread(img, pilmode="RGB")
    imag = Image.fromarray(imag)
    imag = IMAGE_NORMALIZER(imag)
    imag = np.array(imag)
    imag = np.expand_dims(imag, axis=0)  # 增加维度，变成1*C*H*W，方便送入VGG

    inputs = torch.autograd.Variable(torch.from_numpy(imag).float())

    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()

    hist = model(inputs)[pick_layer]  # 得到预处理后的图像的输出特征
    hist = np.sum(hist.data.cpu().numpy(), axis=0)
    hist /= np.sum(hist) + 1e-15  # normalize

    query = {'img': img, 'cls': None, 'hist': hist}
    _, results = infer(query, mode1, samples=samples, lsh=lsh, depth=depth, d_type=d_type)

    # print(results)
    ranked_list = []
    for e in results:
        ranked_list.append(e['img'].split('/')[-1][:-4])

    with open('ranked_list.txt', 'w', encoding='UTF-8') as f:
        for img_name in ranked_list:
            f.write("{}\n".format(img_name))

    print(ranked_list)
    # for i, e in enumerate(results):
    #     img_path = e['img']
    #     image = imageio.imread(img_path, pilmode="RGB")
    #
    #     plt.subplot(2, 5, i+1)
    #     plt.imshow(image)
    #
    # plt.show()
