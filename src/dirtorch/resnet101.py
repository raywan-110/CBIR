from utils import common
import nets
import torch
import torch.nn as nn


checkpoint = "..\\..\\model\\Resnet-101-AP-GeM.pt"


def load_model(path, iscuda):
    checkpoint = common.load_checkpoint(path, iscuda)
    net = nets.create_model(pretrained="", **checkpoint['model_options'])
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    # if 'pca' in checkpoint:
    #     net.pca = checkpoint.get('pca')
    return net


x = torch.zeros((5, 3, 320, 320))

net = load_model(checkpoint, False)
# print(net(x).shape)
net = nn.Sequential(*list(net.children())[:-1])

out = net(x)
out = out.squeeze(-1).squeeze(-1)
print(out.shape)
