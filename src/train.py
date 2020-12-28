from triplet_net import VGGNet
from hard_triplet_loss import HardTripletLoss
from dataset import TripletDataset, AlignCollate
import torch

data_csv = 'simple_data.csv'
BATCH_SIZE = 16
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
N_EPOCH = 4
MODEL_SAVE_PATH = 'trained_model/model_simple.pth'
# LOAD_MODEL_PATH = 'trained_model/model.pth'
LOAD_MODEL_PATH = None

IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CRITERION = HardTripletLoss(margin=0.1, hardest=True, squared=True)

train_dataset = TripletDataset(data_csv=data_csv)
train_align_collate = AlignCollate(IMAGE_HEIGHT, IMAGE_WIDTH, MEAN, STD)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           collate_fn=train_align_collate)

base_model = VGGNet(load_model_path=LOAD_MODEL_PATH, model='vgg19', requires_grad=True)

base_model.fit(CRITERION, LEARNING_RATE, WEIGHT_DECAY, train_loader, N_EPOCH, MODEL_SAVE_PATH)
