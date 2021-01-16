from PyQt5.QtCore import *
from PyQt5.Qt import *
from PyQt5 import QtWidgets
import PyQt5.QtGui as qg
from PyQt5.QtWidgets import QApplication, QMessageBox, QMainWindow, QDialog, QFileDialog, QLabel, QVBoxLayout
from PyQt5.QtGui import *
from PyQt5.uic import loadUi
import os
import torch.nn as nn
from DB import Database
# from network import VGGNet
from model import ModelFeat
from retrieval import search
from dirtorch.utils import common
import dirtorch.nets as nets
import time
import re
from PIL import Image
import imageio
import numpy as np


def load_model(path, iscuda):
    checkpoint = common.load_checkpoint(path, iscuda)
    net = nets.create_model(pretrained="", **checkpoint['model_options'])
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    # if 'pca' in checkpoint:
    #     net.pca = checkpoint.get('pca')
    return net


# DATABASE = Database()
CHECKPOINT = "../model/Resnet-101-AP-GeM.pt"


class Status(QMainWindow):
    def __init__(self):
        super(Status, self).__init__()
        self.img_path = None
        self.method = ModelFeat
        # d: database, o: oxford5k. 先全部读取
        self.index_d, self.dicbase_d, _ = self.method.make_samples(
            db=Database(db_dir='..\\database', db_csv='..\\database.csv'), mode="Linear", is_Oxford=False)
        print('d:', len(self.dicbase_d))
        self.index_o, self.dicbase_o, _ = self.method.make_samples(
            db=Database(db_csv='..\\oxbuild_images.csv', db_dir='..\\oxbuild_images'), mode="Linear", is_Oxford=True)
        print('o:', len(self.dicbase_o))
        # self.model = VGGNet(load_model_path=LOAD_MODEL_PATH, requires_grad=False)
        self.model = load_model(CHECKPOINT, False)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        loadUi(os.path.join("ui\\app.ui"), self)
        self._createLabel()
        self.banner1.dropDown.connect(self.ondropDown)
        self.uploadButton.clicked.connect(self.upload)
        self.retrieveButton.clicked.connect(self.retrieve)
        self.child = Result()
        self.setFixedSize(self.width(), self.height())

    def paintEvent(self, event):
        painter = QPainter(self)
        #todo 1 设置背景颜色
        # painter.setBrush(Qt.green)
        # painter.drawRect(self.rect())

        # #todo 2 设置背景图片，平铺到整个窗口，随着窗口改变而改变
        pixmap = QPixmap(".\\ui\\deer.jpg")
        painter.drawPixmap(self.rect(), pixmap)

    def upload(self):
        img_path = QFileDialog.getOpenFileName(self, "上传图片", ".",
                                               "图片文件(*.png *.jpg *.tiff *.pjp *.jfif *.jpeg *.svg)")
        self.img_path = os.path.join(*img_path[0].split("/"))
        self.img_path = os.path.normpath(self.img_path)

        self.img_path = self.img_path[:2] + '\\' + self.img_path[2:]  # 磁盘冒号后加'\\'

        self.imgpathEdit.setText(self.img_path)
        jpg = QPixmap(self.img_path)
        self.banner1.setPixmap(jpg)
        # # print(img_path[0])
        # # print(self.img_path)
        # print(self.img_path)
        # print(os.path.exists(self.img_path))

    def retrieve(self):
        if self.img_path is None:
            msg_box = QMessageBox(QMessageBox.Warning, "警告", "请先上传图片")
            msg_box.exec_()
        else:
            t = time.time()
            print(self.comboBox.currentText())
            if self.comboBox.currentText() == "oxford5k":
                results = search(self.model, self.img_path, self.index_o, self.dicbase_o)
            elif self.comboBox.currentText() == "database":
                results = search(self.model, self.img_path, self.index_d, self.dicbase_d)
            print(time.time() - t)
            print(results)
            ranked_list = []
            for d in results:
                ranked_list.append(os.path.join(*d.split("/")))
            print(os.path.exists(ranked_list[0]))
            self.child.result_list = ranked_list
            self.child.show()  # 模态窗口
            self.child.show_res()

    def _createLabel(self):
        self.banner1 = DropLabel(self)
        self.banner1.setFrameShape(QtWidgets.QFrame.Box)
        self.banner1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.banner1.setLineWidth(5)
        self.banner1.setStyleSheet('background-color: rgb(255, 248, 238)')
        self.banner1.setGeometry(QRect(60, 80, 400, 300))  # (x, y, width, height)
        self.banner1.setObjectName("banner1")
        self.banner1.setText('将图片拖拽进来试试看?')
        self.banner1.setFont(QFont("Roman times",15,QFont.Bold))
        self.banner1.setAlignment(Qt.AlignCenter)
        self.banner1.move(300,200)

    def ondropDown(self, _label, _path):
        if _path.endswith('.jpg' or 'png'):
            self.img_path = _path
            # pixmap = QPixmap(_path)
            print(_path)
            ima = imageio.imread(_path)
            new_img = self.img_resize(ima, max_h=800, max_w=400)
            show_path = os.path.join("tmp", "origin.jpg")
            imageio.imsave(show_path, new_img)
            # _label.setScaledContents(True)  # 自适应大小
            _label.setPixmap(QPixmap(show_path))  # 显示在控件上

    def img_resize(self, img, max_h, max_w):
        ima = Image.fromarray(img)
        h, w = ima.size
        if h > max_h or w > max_w:
            rate = max(h / max_h, w / max_w)
            new_h, new_w = int(h / rate), int(w // rate)
            new_ima = ima.resize((new_h, new_w))
            img = np.array(new_ima)
        return img


class Result(QDialog):
    def __init__(self):
        super(Result, self).__init__()
        loadUi(os.path.join("ui\\result.ui"), self)
        self.setModal(True)
        self.setWindowModality(Qt.ApplicationModal)  # 不关闭子窗口不可显示
        self.setWindowTitle('result')
        self.setWindowIcon(qg.QIcon('ui\\favicon.ico'))
        self.result_list = []
        self.setFixedSize(1600, 800)

    def show_res(self):
        for i, img in enumerate(self.result_list):
            ima = imageio.imread(img)
            new_img = self.img_resize(ima, max_h=800, max_w=400)
            show_path = os.path.join("tmp", "{}.jpg".format(i))
            imageio.imsave(show_path, new_img)
            exec('self.label_{}.setPixmap(QPixmap(show_path))'.format(i + 1))

    def img_resize(self, img, max_h, max_w):
        ima = Image.fromarray(img)
        h, w = ima.size
        if h > max_h or w > max_w:
            rate = max(h / max_h, w / max_w)
            new_h, new_w = int(h / rate), int(w // rate)
            new_ima = ima.resize((new_h, new_w))
            img = np.array(new_ima)
        return img


class DropLabel(QLabel):
    # 设置自定义信号函数在使用时默认传递两个参数
    dropDown = pyqtSignal(object, str)
    def __init__(self, *args, **kwargs):
        QLabel.__init__(self, *args, **kwargs)
        self.setAcceptDrops(True)

    # 定义dragEnterEvent来接收图片
    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        super(DropLabel, self).dropEvent(event)
        image = event.mimeData().text()
        image_path = re.sub('^file:///', '', image)
        # 槽函数，返回DropLabel对象和路径
        self.dropDown.emit(self, image_path)  # 发射信号 
        event.acceptProposedAction()


if __name__ == '__main__':
    app = QApplication([])
    stats = Status()
    stats.setWindowTitle('CBIR system')
    stats.setWindowIcon(qg.QIcon('ui\\favicon.ico'))
    stats.show()
    app.exec()
