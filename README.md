# Dive-Into-Deep-Learning-PyTorch-PDF
<div align=center>
<img width="500" src="img/cover.png" alt="封面"/>
</div>

## 简介
&emsp;&emsp;本项目对中文版《动手学深度学习》（第一版）中的代码进行整理，并参考一些优秀的GitHub项目给出基于PyTorch的实现方法。为了方便阅读，本项目给出[全书PyTorch版的PDF版本](https://github.com/wzy6642/Dive-Into-Deep-Learning-PyTorch-PDF/tree/main/pdf)。欢迎大家Download，Star，Fork。除了原书内容外，我们还为每一章增加了本章附录，用于对该章节中用到的函数以及数学计算加以说明，除此之外还增加了语义分割网络（U-Net）的实现。书籍百度云链接：https://pan.baidu.com/s/1l8yDHVcB0FXPLH1nL542xA 密码：euqd       
&emsp;&emsp;原书作者：阿斯顿·张、李沐、扎卡里 C. 立顿、亚历山大 J. 斯莫拉以及其他社区贡献者。       
&emsp;&emsp;备注: d2lzh.py与其它代码需要放到同一个文件夹下。

## 目录
* 1\. 预备知识
   * 1.1 数据操作
   * 1.2 自动求梯度
   * 1.3 查阅文档
   * 1.4 本章附录
* 2\. 深度学习基础
   * 2.1 线性回归
   * 2.2 线性回归的从零开始实现
   * 2.3 线性回归的简洁实现
   * 2.4 softmax回归
   * 2.5 图像分类数据集（Fashion-MNIST）
   * 2.6 softmax回归的从零开始实现
   * 2.7 softmax回归的简洁实现
   * 2.8 多层感知机
   * 2.9 多层感知机的从零开始实现
   * 2.10 多层感知机的简洁实现
   * 2.11 模型选择、欠拟合和过拟合
   * 2.12 权重衰减
   * 2.13 丢弃法
   * 2.14 正向传播、反向传播和计算图
   * 2.15 数值稳定性和模型初始化
   * 2.16 实战Kaggle比赛：房价预测
   * 2.17 本章附录
* 3\. 深度学习计算
   * 3.1 模型构造
   * 3.2 模型参数的访问、初始化和共享
   * 3.3 自定义层
   * 3.4 读取和存储
   * 3.5 GPU计算
   * 3.6 本章附录
* 4\. 卷积神经网络
   * 4.1 二维卷积层
   * 4.2 填充和步幅
   * 4.3 多输入通道和多输出通道
   * 4.4 池化层
   * 4.5 卷积神经网络（LeNet）
   * 4.6 深度卷积神经网络（AlexNet）
   * 4.7 使用重复元素的网络（VGG）
   * 4.8 网络中的网络（NiN）
   * 4.9 含并行连结的网络（GoogLeNet）
   * 4.10 批量归一化
   * 4.11 残差网络（ResNet）
   * 4.12 稠密连接网络（DenseNet）
   * 4.13 本章附录
* 5\. 循环神经网络
   * 5.1 语言模型
   * 5.2 循环神经网络
   * 5.3 语言模型数据集（周杰伦专辑歌词）
   * 5.4 循环神经网络的从零开始实现
   * 5.5 循环神经网络的简洁实现
   * 5.6 通过时间反向传播
   * 5.7 门控循环单元（GRU）
   * 5.8 长短期记忆（LSTM）
   * 5.9 深度循环神经网络
   * 5.10 双向循环神经网络
   * 5.11 本章附录
* 6\. 优化算法
   * 6.1 优化与深度学习
   * 6.2 梯度下降和随机梯度下降
   * 6.3 小批量随机梯度下降
   * 6.4 动量法
   * 6.5 AdaGrad算法
   * 6.6 RMSProp算法
   * 6.7 AdaDelta算法
   * 6.8 Adam算法
   * 6.9 本章附录
* 7\. 计算性能
   * 7.1 命令式和符号式混合编程
   * 7.2 自动并行计算
   * 7.3 多GPU计算
   * 7.4 本章附录
* 8\. 计算机视觉
   * 8.1 图像增广
   * 8.2 微调
   * 8.3 目标检测和边界框
   * 8.4 锚框
   * 8.5 多尺度目标检测
   * 8.6 目标检测数据集（皮卡丘）
   * 8.7 单发多框检测（SSD）
   * 8.8 区域卷积神经网络（R-CNN）系列
   * 8.9 语义分割和数据集
   * 8.10 全卷积网络（FCN）
   * 8.11 样式迁移
   * 8.12 实战Kaggle比赛：图像分类（CIFAR-10）
   * 8.13 实战Kaggle比赛：狗的品种识别（ImageNet Dogs）
   * 8.14 语义分割网络（U-Net）
   * 8.15 本章附录
* 9\. 自然语言处理
   * 9.1 词嵌入（word2vec）
   * 9.2 近似训练
   * 9.3 word2vec的实现
   * 9.4 子词嵌入（fastText）
   * 9.5 全局向量的词嵌入（GloVe）
   * 9.6 求近义词和类比词
   * 9.7 文本情感分类：使用循环神经网络
   * 9.8 文本情感分类：使用卷积神经网络（textCNN）
   * 9.9 编码器—解码器（seq2seq）
   * 9.10 束搜索
   * 9.11 注意力机制
   * 9.12 机器翻译
   * 9.13 本章附录
## 环境
matplotlib==3.3.2       
torch==1.1.0       
torchvision==0.3.0       
torchtext==0.4.0       
CUDA Version==11.0


## 参考
本书PyTorch实现：[Dive-into-DL-PyTorch](https://github.com/ShusenTang/Dive-into-DL-PyTorch)       
本书TendorFlow2.0实现：[Dive-into-DL-TensorFlow2.0](https://github.com/TrickyGo/Dive-into-DL-TensorFlow2.0)

## 原书地址
中文版：[动手学深度学习](https://zh.d2l.ai/) | [Github仓库](https://github.com/d2l-ai/d2l-zh)       
English Version: [Dive into Deep Learning](https://d2l.ai/) | [Github Repo](https://github.com/d2l-ai/d2l-en)

## 引用
如果您在研究中使用了这个项目请引用原书:
```
@book{zhang2019dive,
    title={Dive into Deep Learning},
    author={Aston Zhang and Zachary C. Lipton and Mu Li and Alexander J. Smola},
    note={\url{http://www.d2l.ai}},
    year={2020}
}
```
