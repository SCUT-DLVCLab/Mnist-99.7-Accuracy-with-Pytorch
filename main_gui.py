import os,sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog
from Ui_widget import Ui_AreYouSmling
from Mydataset import get_test_data_loader
from PIL import Image,ImageQt

import torchvision.transforms as transforms
import torch
import numpy as np
from cnnModel import CNN
from Predict import predict
import matplotlib.pyplot as plt

class mywindow(QtWidgets.QWidget,Ui_AreYouSmling):
    def __init__(self):
        super(mywindow,self).__init__()
        self.cwd=os.getcwd()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.read_dataset)
        self.pushButton_2.clicked.connect(self.read_file)
        self.pushButton_3.clicked.connect(self.predict)
    def read_file(self):
        self.label_2.setText(" ")
        self.label_3.setText(" ")
        file,filetype=QFileDialog.getOpenFileName(self,'open image',self.cwd,"*.JPG,*.JPEG,*.png,*.jpg,ALL Files(*)")
        if not file=='':
            self.image=Image.open(file).convert('L')
            jpg = QtGui.QPixmap(file).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(jpg)

    def read_dataset(self):
        self.label_2.setText(" ")
        batch=get_test_data_loader(1)
        image,label=next(iter(batch))
        self.image=image #将tensor传走
        self.label_3.setText(str(label.numpy()))
        image=image.squeeze(dim=0)
        unloader=transforms.ToPILImage()
        image=unloader(image)
        pixmap=ImageQt.toqpixmap(image)
        jpg=QtGui.QPixmap(pixmap).scaled(self.label.width(),self.label.height())
        self.label.setPixmap(jpg)
    
       
    def predict(self):
        #先将图片转为PIL形式
        
        image=self.image
        pred=predict(image)
        self.label_4.setText('Predicting')
        #pred=predict(image)
        self.label_4.setText('Predicted')
        self.label_2.setText(str(pred.numpy().item()))


if __name__=="__main__":
    app=QtWidgets.QApplication(sys.argv)
    myshow=mywindow()
    myshow.show()
    sys.exit(app.exec_())