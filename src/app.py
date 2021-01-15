from PyQt5.QtCore import *
from PyQt5.Qt import *
from PyQt5 import QtWidgets 
import PyQt5.QtGui as qg
from PyQt5.QtWidgets import QApplication, QMessageBox, QMainWindow, QDialog, QFileDialog, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
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


def load_model(path, iscuda):
    checkpoint = common.load_checkpoint(path, iscuda)
    net = nets.create_model(pretrained="", **checkpoint['model_options'])
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    # if 'pca' in checkpoint:
    #     net.pca = checkpoint.get('pca')
    return net


DATABASE = Database()
CHECKPOINT = "../model/Resnet-101-AP-GeM.pt"


class Status(QMainWindow):
    def __init__(self):
        super(Status, self).__init__()
        self.img_path = None
        self.method = ModelFeat
        self.index, self.dicbase, _ = self.method.make_samples(db=DATABASE, mode="Linear")
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
            results = search(self.model, self.img_path, self.index, self.dicbase)
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
        self.banner1 =DropLabel(self)
        self.banner1.setFrameShape(QtWidgets.QFrame.Box)
        self.banner1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.banner1.setLineWidth(5)
        self.banner1.setStyleSheet('background-color: rgb(192, 192, 192)')
        self.banner1.setGeometry(QRect(60, 80, 400, 300))  # (x, y, width, height)
        self.banner1.setObjectName("banner1")
        self.banner1.setText('将图片拖拽到此区域')
        self.banner1.setAlignment(Qt.AlignCenter)
        self.banner1.move(300,150)

    def ondropDown(self, _label, _path):
        if _path.endswith('.jpg'):
            self.img_path = _path
            pixmap = QPixmap(_path)
            print(_path)
            _label.setScaledContents(True)  # 自适应大小
            _label.setPixmap(pixmap)  # 显示在控件上


class Result(QDialog):
    def __init__(self):
        super(Result, self).__init__()
        loadUi(os.path.join("ui\\result.ui"), self)
        self.setModal(True)
        self.setWindowModality(Qt.ApplicationModal)  # 不关闭子窗口不可显示
        self.setWindowTitle('result')
        self.setWindowIcon(qg.QIcon('ui\\favicon.ico'))
        # self.label_1.setScaledContents(True)
        # self.label_2.setScaledContents(True)
        # self.label_3.setScaledContents(True)
        # self.label_4.setScaledContents(True)
        # self.label_5.setScaledContents(True)
        # self.label_6.setScaledContents(True)
        # self.label_7.setScaledContents(True)
        # self.label_8.setScaledContents(True)
        # self.label_9.setScaledContents(True)
        # self.label_10.setScaledContents(True)
        self.result_list = []
        self.setFixedSize(self.width(), self.height())

    def show_res(self):
        for i, img in enumerate(self.result_list):
            exec('self.label_{}.setPixmap(QPixmap(img))'.format(i + 1))

class DropLabel(QLabel):
    # 设置自定义信号函数在使用时默认传递两个参数
    dropDown = pyqtSignal(object,str)

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
