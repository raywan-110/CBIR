from evaluate import infer
from DB import Database
import time
import numpy as np
from vggnet import VGGNetFeat, VGGNet
from resnet import ResNetFeat
import imageio
import torch
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *

means = np.array([103.939, 116.779, 123.68]) / 255

depth = 10
d_type = 'cosine'
query_idx = 10
mode1 = 'LSH'
VGG_model = 'vgg19'  # model type
pick_layer = 'avg'  # extract feature of this layer
feat_dim = 512  # 输出特征的维度

# img = 'D:\\Users\\15657\\Desktop\\image_0003_flip.jpg'

if __name__ == '__main__':
    db = Database()
    img = filedialog.askopenfilename()

    # retrieve by VGG
    method = VGGNetFeat()
    samples, lsh = method.make_samples(db, mode1)

    model = VGGNet(requires_grad=False, model=VGG_model)
    model.eval()

    imag = imageio.imread(img, pilmode="RGB")
    imag = imag[:, :, ::-1]  # switch to BGR 第三个维度倒序
    imag = np.transpose(imag, (2, 0, 1)) / 255.  # 转置成C*H*W
    imag[0] -= means[0]  # reduce B's mean
    imag[1] -= means[1]  # reduce G's mean
    imag[2] -= means[2]  # reduce R's mean
    imag = np.expand_dims(imag, axis=0)  # 增加维度，变成1*C*H*W，方便送入VGG

    inputs = torch.autograd.Variable(torch.from_numpy(imag).float())

    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()

    hist = model(inputs)[pick_layer]  # 得到预处理后的图像的输出特征
    hist = np.sum(hist.data.cpu().numpy(), axis=0)
    hist /= np.sum(hist)  # normalize

    query = {'img': img, 'cls': None, 'hist': hist}
    _, results = infer(query, mode1, samples=samples, lsh=lsh, depth=depth, d_type=d_type)

    for i, e in enumerate(results):
        img_path = e['img']
        image = imageio.imread(img_path, pilmode="RGB")

        plt.subplot(2, 5, i+1)
        plt.imshow(image)

    plt.show()
