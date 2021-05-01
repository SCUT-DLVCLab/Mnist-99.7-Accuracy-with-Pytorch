## 简介
改代码设计了一个四层卷积神经网络用以识别MNIST数据集，最终训练的得的结果中，单个网络在测试集上可到到最高`99.67%`的识别率，通过`级联九个神经网络`，可以使得测试集达到`99.7%`的准确率
![Image text](https://raw.githubusercontent.com/Mountchicken/Mnist-99.7-Accuracy-with-Pytorch/main/Imagesforgithub/network.jpg)
该网络体积小，30个epochs的训练可以达到99.67%的准确率，耗时4分钟左右。同时在代码中集成了tensorboard调参方法以及一个由PyQt5设计的APP

## 项目结构
### 文件

- `cnnModel.py`: 定义模型
- `Mydataset.py`:定义数据集以及dataloader，初次运行时请修改其中的`DATASET_PATH`以读取MNIST数据集
- `Train.py`: 训练模型，初次运行时请修改其中的`SAVED_PATH`指定模型保存地址
- `Test.py`: 测试模型在训练集准确率,初次运行时请修改其中的`path`指定模型地址。调用`SingleTest():`进行单模型测试，调用`CombinationTest():`进行模型级联测试
- `ConfusuinMatrix.py`: 绘制单个模型的混淆矩阵
- `Predict.py`: 用以预测单幅图像
- `main_gui.py,widget.ui,ui_widget.py`: GUI封装代码

### 文件夹
- `Trained_models`: 存放预训练的模型参数
- `datasets`: 存放MNIST数据集
- `Images`: 存放自己的手写数字图片


## 如何使用

### 如何预测
-`Predict.py`文件中，line24选择测图片即可

### 如何使用tensorboard进行调参训练
![Image text](https://raw.githubusercontent.com/Mountchicken/Mnist-99.7-Accuracy-with-Pytorch/main/Imagesforgithub/tensorboard.jpg)
- `pip install tensorboard`
- 在`Train.py`中修改 `TestParameters`，往列表中加入训练参数，代码会自动对所有可能的训练参数进行训练
- 训练结束后，在Anaconda Prompt中执行命令行 ` tensorboard --logdir=runs`

### 如何启动GUI
![Image text](https://raw.githubusercontent.com/Mountchicken/Mnist-99.7-Accuracy-with-Pytorch/main/Imagesforgithub/GUI.jpg)
- `pip install PyQt5`
- `pip install PyQt5-tools`
- 运行文件`main_gui.py`


## 联系方式
- mountchicken@outlook.com


