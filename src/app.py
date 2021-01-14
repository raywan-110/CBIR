from PyQt5.QtWidgets import QApplication, QMessageBox, QMainWindow, QDialog, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QFile
from PyQt5.uic import loadUi
import os
from DB import Database
from network import VGGNet
from model import ModelFeat
from retrieval import search

DATABASE = Database()
LOAD_MODEL_PATH = None


class Status(QMainWindow):
    def __init__(self):
        super(Status, self).__init__()
        self.img_path = None
        self.method = ModelFeat
        self.index, self.dicbase, _ = self.method.make_samples(db=DATABASE, mode="Linear")
        self.model = VGGNet(load_model_path=LOAD_MODEL_PATH, requires_grad=False)
        self.model.eval()

        loadUi(os.path.join("ui\\app.ui"), self)

        self.uploadButton.clicked.connect(self.upload)
        self.retrieveButton.clicked.connect(self.retrieve)

        self.child = Result()

    def upload(self):
        img_path = QFileDialog.getOpenFileName(self, "上传图片", ".",
                                               "图片文件(*.png *.jpg *.tiff *.pjp *.jfif *.jpeg *.svg)")
        self.img_path = os.path.join(*img_path[0].split("/"))
        self.img_path = os.path.normpath(self.img_path)

        self.img_path = self.img_path[:2] + '\\' + self.img_path[2:]  # 磁盘冒号后加'\\'

        self.imgpathEdit.setText(self.img_path)
        jpg = QPixmap(self.img_path)
        self.label_in.setPixmap(jpg)
        # # print(img_path[0])
        # # print(self.img_path)
        # print(self.img_path)
        # print(os.path.exists(self.img_path))

    def retrieve(self):
        if self.img_path is None:
            msg_box = QMessageBox(QMessageBox.Warning, "警告", "请先上传图片")
            msg_box.exec_()
        else:
            results = search(self.model, self.img_path, self.index, self.dicbase)
            print(results)
            ranked_list = []
            for d in results:
                ranked_list.append(os.path.join("..", *d.split("/")))
            print(ranked_list)
            self.child.result_list = ranked_list
            self.child.show()
            self.child.show_res()


class Result(QDialog):
    def __init__(self):
        super(Result, self).__init__()
        loadUi(os.path.join("ui\\result.ui"), self)
        self.result_list = []

    def show_res(self):
        for i, img in enumerate(self.result_list):
            exec('self.label_{}.setPixmap(QPixmap(img))'.format(i + 1))


if __name__ == '__main__':
    app = QApplication([])
    stats = Status()
    stats.show()
    app.exec()
