from DB import Database
import numpy as np
from network import VGGNet
from model import ModelFeat
import imageio
import torch
import torchvision.transforms as transforms
from PIL import Image

DEPTH = 10
D_TYPE = 'd2'
mode1 = 'Linear'
PICK_LAYER = 'avg'  # extract feature of this layer
# LOAD_MODEL_PATH = 'trained_model/model_1.pth'
LOAD_MODEL_PATH = None

IMAGE_NORMALIZER = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def search(model, img_path, index, dicbase, pick_layer='avg'):
    imag = imageio.imread(img_path, pilmode="RGB")
    imag = Image.fromarray(imag)
    imag = IMAGE_NORMALIZER(imag)
    imag = np.array(imag)
    imag = np.expand_dims(imag, axis=0)  # 增加维度，变成1*C*H*W，方便送入VGG

    inputs = torch.autograd.Variable(torch.from_numpy(imag).float())
    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     inputs = inputs.cuda()
    # else:
    model.cpu()
    inputs.cpu()

    hist = model(inputs)[pick_layer]  # 得到预处理后的图像的输出特征
    hist = np.sum(hist.data.cpu().numpy(), axis=0)
    hist /= np.sum(hist) + 1e-15  # normalize

    hist = hist.reshape(1, -1)
    result_D, result_I = index.search(hist, 10 + 1)
    result_D = result_D[:, 1:]
    result_I = result_I[:, 1:]
    result_I = result_I.reshape(-1)

    img_path = dicbase.iloc[:, 1].values

    results = img_path[result_I]
    return results


if __name__ == '__main__':
    db = Database()
    img = '../oxbuild_images/all_souls_000013.jpg'
    # img = 'database/chair/image_0001.jpg'
    method = ModelFeat
    index, dicbase, vecbase = method.make_samples(db, mode1)

    model = VGGNet(load_model_path=LOAD_MODEL_PATH, requires_grad=False)
    model.eval()
    results = search(model, img, index, dicbase, pick_layer='avg')

    print(results)
    ranked_list = []
    for e in results:
        ranked_list.append(e.split('/')[-1][:-4])

    with open('ranked_list.txt', 'w', encoding='UTF-8') as f:
        for img_name in ranked_list:
            f.write("{}\n".format(img_name))

    # print(ranked_list)
    # for i, e in enumerate(results):
    #     img_path = e['img']
    #     image = imageio.imread(img_path, pilmode="RGB")
    #
    #     plt.subplot(2, 5, i+1)
    #     plt.imshow(image)
    #
    # plt.show()
