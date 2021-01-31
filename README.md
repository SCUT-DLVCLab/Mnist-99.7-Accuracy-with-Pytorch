# Mnist-99.7-Accuracy-with-Pytorch
# 16PSK通信系统

## 系统简介
2020-2021年秋季学期华南理工大学本科生课程《信息论基础与通信原理》大作业/Big Project of the 2020-2021 SCUT Course "Information Theory and Principle of Communication"

本系统采用`A律PCM`编码，调制方式选用`16PSK`，解调方式为`相关解调`和`相干解调`
### **给参考者的警告**
- **虽然本项目完成了课程设计的全部要求，我们自身验证也感觉“挺好”，但是根据期末总成绩推算老师给这个作业的分数非常低！！！初步估计是我们的差错控制编码模块效果不太理想，误码率并没有得到显著提升(看到有的组误码率下降了10倍之多，但我们只有零点几)，同时根据该模型绘制出的误码率-比特能量图像与理论存在较严重的偏差，建议“参考”时重新编写该模块的代码。**
- **(希望老师看不到...)和其他组交流了一下，感觉这个作业的关键是课设报告！！！期末上面催命出成绩老师也没有什么时间仔细看，所以把报告写的好看一点，排版漂亮一点很重要...有的组实际上没有完成一些指标要求，但是报告做的比较好所以最后成绩非常高...**

## 项目结构
### 文件
- `main.py`: 实现信号`audio.wav`在通信系统中的传输，**验证时请运行此文件**，运行后可得到接收信号`audio_correlated_decoded.wav`或`audio_coherent_decoded.wav`以及运行结果数据`correlated.txt`或`coherent.txt`
- `test_and_plot`: 测试`audio.wav`中少量数据点的传输效果，并绘制图像
- `performance_estimation`: 测试系统输出的误差
- `audio.wav`: 测试信号，选用歌曲《歌唱祖国》

### 文件夹
- `module`: 存放自定义模块
  - `audio_func.py`: 信号基本操作，包括wav文件播放和ndarray到wav的转换
  - `channel.py`: 信道模拟，为信号加入AWGN
  - `pcm.py`: 实现A律PCM编译码功能
  - `psk16.py`: 实现信号的16PSK调制解调
- `result`: 存放程序运行结果
  - `correlated.txt`:相关解调结果，第一行为误比特个数，第二行为误比特个数与信号均值之比，第三行为运行时间
  - `coherent.txt`: 相干解调结果，第一行为比特个数，第二行为误比特个数与信号均值之比，第三行为运行时间
  - `audio_correlated_decoded.wav`: 系统使用相关解调时的输出
  - `audio_coherent_decoded.wav`：系统使用相干解调时的输出
  - `audio_correlated_decoded_4.wav`: 系统使用相关解调时的输出，噪声增强
  - `audio_coherent_decoded.wav`：系统使用相干解调时的输出，噪声增强
- `figure`: 存放输出的图像
  - `original_signal.png`: 输入信号以及两种解调方式的输出
  - `PCM_encoded_signal`: PCM编码后的信号，以及经过相关/相干解调后的PCM编码
  - `16PSK_modulated_signal`: 经过16PSK调制后的模拟信号波形
  - `corr_snr_err`: 有差错控制编码时误码率与信噪比的关系
  - `snr_err`: 无差错控制编码时误码率与信噪比的关系


## **Warnings of the Program**
- 作者编写程序时使用了Visual Studio Code，因此文件目录中含有配置文件夹`.vscode`。请按照您的工作环境修改或删除其中的内容。
- 16PSK调制解调的算法本身具有一定复杂度，因此运行`main.py`将十分耗费时间。在作者的工作环境下(Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz, 24.0GB RAM)进行一种调制解调需花费30-40分钟。
- 由于程序运行开销较大，因此我们在`result`文件夹中存放了两种解调方式的输出`audio_correlated_decoded.wav`和`audio_coherent_decoded.wav`，您可以直接播放这两个音频文件来检验系统效果。**我们保证这两个文件是经过程序运行而得到的真实结果**。
- 如果默认的测试信号（3分多钟）给您的计算机造成过大负担，可尝试将测试信号截短或者更换其他单通道wav信号。
- `psk16.psk16_modulate` `psk16.psk16_correlated_demodulate` `psk16.psk16_coherent_demodulate`中的fc,fs为调制载波频率和码元速率，这里我们默认其为$f_c=10f_s$，您可以对之进行修改，但要保证它们的倍数为整数；我们不建议您将倍数设置得过大，否则运算量将成倍增加。
- 计算机只能使用离散信号代表连续的模拟信号，`psk16.psk16_modulate` `psk16.psk16_correlated_demodulate` `psk16.psk16_coherent_demodulate`中最后一个参数控制一个模拟载波周期中包含的数据点数（整数），请不要将该值设得过大，否则运算量将成倍增加。

## **环境依赖**
以下为作者编写程序时的工作环境，仅供参考
- pyAudio 0.2.11
- numpy 1.19.1
- scipy 1.5.2
- matplotlib 3.3.1

## 作者
*华南理工大学电子与信息学院2018级信息工程创新班*   
- 林泽柠 zening.lin@outlook.com  
- 李昱澍  
- 谭启恒  
- 王庆丰 
