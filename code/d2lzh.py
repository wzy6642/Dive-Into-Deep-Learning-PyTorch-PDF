# -*- coding: utf-8 -*-
"""
Created on Fri Sep 4 11:25:40 2020
@function：
    动手学深度学习中常用的函数
    
@reference：

@note：

@author: Zhenyu Wu
"""
import matplotlib.pyplot as plt
from IPython import display
import torch
from torch import nn
import random
import sys
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import zipfile
import random
import math
import numpy as np
import warnings
import os
import json
from PIL import Image
from torch import optim
from tqdm import tqdm
import shutil
import collections
import torchtext.vocab as Vocab
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


def use_svg_display():
    """
    function：
        用矢量图显示
        
    Parameters:

    Returns:

    Modify:
        2020-11-25
    """
    display.set_matplotlib_formats('svg')

    
def set_figsize(figsize=(3.5, 2.5)):
    """
    function：
        设置图的尺寸
        
    Parameters:
        figsize - 图框大小(tuple)

    Returns:

    Modify:
        2020-11-25
    """
    plt.rcParams['figure.figsize'] = figsize
    
    
def data_iter(batch_size, features, labels):
    """
    function：
        将数据集按照最小批的大小进行分割
        
    Parameters:
        batch_size - 最小批的大小(int)
        features - 特征(Tensor)
        labels - 标签(Tensor)

    Returns:
        特征，标签

    Modify:
        2020-11-25
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    # 样本的读取顺序是随机的
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        # 最后一次可能不足一个batch
        j = torch.LongTensor(indices[i: min(i+batch_size, num_examples)])
        # index_select函数根据索引返回对应元素
        yield features.index_select(0, j), labels.index_select(0, j)
        
        
def linreg(X, w, b):
    """
    function：
        线性回归模型
        
    Parameters:
        X - 特征矩阵(Tensor)
        w - 系数矩阵(Tensor)
        b - 偏置张量(Tensor)

    Returns:
        线性回归模型：y=X*w+b

    Modify:
        2020-11-26
    """
    return torch.mm(X, w) + b


def squared_loss(y_hat, y):
    """
    function：
        MSE损失函数
        
    Parameters:
        y_hat - 模型预测值(Tensor)
        y - 真实标签(Tensor)

    Returns:
        损失函数计算结果

    Modify:
        2020-11-26
    """
    return (y_hat - y.view(y_hat.size())) ** 2 / 2 


def sgd(params, lr, batch_size):
    """
    function：
        SGD优化器
        
    Parameters:
        params - 模型的参数(Tensor)
        lr - 学习率(float)
        batch_size - 最小批大小(int)

    Returns:
        利用梯度下降算法优化超参

    Modify:
        2020-11-26
    """
    for param in params:
        # 注意这里更改param时用的param.data
        param.data -= lr * param.grad / batch_size
        

def get_fashion_mnist_labels(labels):
    """
    function：
        将fashion mnist数据集中的label转为字符串标签
        
    Parameters:
        labels - 数值标签列表(list, int)

    Returns:
        映射为字符串的标签列表

    Modify:
        2020-11-26
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    """
    function：
        对fashion mnist数据集中的样本进行可视化
        
    Parameters:
        images - 图片(list, Tensor)
        labels - 标签(list, str)

    Returns:

    Modify:
        2020-11-26
    """
    use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
        
        
def load_data_fashion_mnist(batch_size, resize=None):
    """
    function：
        将fashion mnist数据集划分为小批量样本
        
    Parameters:
        batch_size - 小批量样本的大小(int)
        resize - 对图像的维度进行扩大

    Returns:
        train_iter - 训练集样本划分为最小批的结果
        test_iter - 测试集样本划分为最小批的结果

    Modify:
        2020-11-26
        2020-12-10 添加图像维度变化
    """
    # 存储图像处理流程
    trans = []
    if resize:
        trans.append(transforms.Resize(size=resize))
    trans.append(transforms.ToTensor())
    transform = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='data/FashionMNIST', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root='data/FashionMNIST', train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        # 0表示不用额外的进程来加速读取数据
        num_workers = 0
    else: 
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter
    
    
def evaluate_accuracy(data_iter, net, device=None):
    """
    function：
        计算多分类模型预测结果的准确率
        
    Parameters:
        data_iter - 样本划分为最小批的结果
        net - 定义的网络
        device - 指定计算在GPU或者CPU上进行

    Returns:
        准确率计算结果

    Modify:
        2020-11-30
        2020-12-03 增加模型训练模型和推理模式的判别
        2020-12-10 增加指定运行计算位置的方法
    """
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        if isinstance(net, torch.nn.Module):
            # 评估模式, 这会关闭dropout
            net.eval()
            # .cpu()保证可以进行数值加减
            acc_sum += (net(X.to(device)).argmax(dim=1).long() == y.to(device).long()).float().sum().cpu().item()
            # 改回训练模式
            net.train()
        # 自定义的模型, 2.13节之后不会用到, 不考虑GPU
        else:
            if ('is_training' in net.__code__.co_varnames):
                # 将is_training设置成False
                acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
            else:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    """
    function：
        利用softmax回归模型对图像进行分类识别
        
    Parameters:
        net - 定义的网络
        train_iter - 训练集样本划分为最小批的结果
        test_iter - 测试集样本划分为最小批的结果
        loss - 损失函数
        num_epochs - 迭代次数
        batch_size - 最小批大小
        params - 参数
        lr - 学习率
        optimizer - 优化器

    Returns:

    Modify:
        2020-11-30
    """
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (epoch+1, train_l_sum/n, train_acc_sum/n, test_acc))
        
        
class FlattenLayer(nn.Module):
    """
    function：
        将张量（batch_size, n, m, y）变为（batch_size, n*m*y）
        
    Parameters:
        x - 维度要被变换的张量

    Returns:
        维度变换之后的张量

    Modify:
        2020-12-02
    """
    def __init__(self):
        super(FlattenLayer, self).__init__()
    # x shape: (batch, *, *, ...)
    def forward(self, x):
        return x.view(x.shape[0], -1)
    

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    """
    function：
        在对数坐标系下绘制训练曲线
        
    Parameters:
        x_vals - x轴的数值
        y_vals - y轴的数值
        x_label - x轴的标签
        y_label - y轴的标签
        x2_vals - 第二个x轴的数值
        y2_vals - 第二个y轴的数值
        legend - 图例
        figsize - 图框大小

    Returns:

    Modify:
        2020-12-03
    """
    plt.figure(figsize=figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # 对数坐标
    plt.semilogy(x_vals, y_vals)
    # 绘制测试集上的曲线
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
        

def corr2d(X, K):
    """
    function：
        二维卷积运算
        
    Parameters:
        X - 待卷积张量(Tensor)
        K - 卷积核(Tensor)

    Returns:
        Y - 二维卷积计算结果(Tensor)

    Modify:
        2020-12-09
    """
    # 行、列值
    h, w = K.shape
    # 卷积结果的存放位置
    Y = torch.zeros((X.shape[0]-h+1, X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y


def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    """
    function：
        利用softmax回归模型对图像进行分类识别
        
    Parameters:
        net - 定义的网络
        train_iter - 训练集样本划分为最小批的结果
        test_iter - 测试集样本划分为最小批的结果
        num_epochs - 迭代次数
        batch_size - 最小批大小
        optimizer - 优化器
        device - 指定计算在GPU或者CPU上进行

    Returns:

    Modify:
        2020-12-10
    """
    # 将模型加载到指定运算器中
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            # 梯度清零
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec' % (epoch+1, train_l_sum/batch_count, train_acc_sum/n, test_acc, time.time()-start))
        
                       
class GlobalAvgPool2d(nn.Module):
    """
    function：
        全局平均池化
        
    Parameters:
        x - 要被池化的张量

    Returns:
        全局平均池化计算结果

    Modify:
        2020-12-10
    """
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])
    
    
class Residual(nn.Module):
    """
    function：
        残差块
        
    Parameters:
        in_channels - 输入通道数
        out_channels - 输出通道数
        use_1x1conv - 是否使用1x1卷积核
        stride = 步长

    Returns:
        残差块计算结果

    Modify:
        2020-12-15
    """
    # 输入通道数、输出通道数、是否使用1x1卷积核、步长
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        # 3x3搭配1步长，特征图大小不变
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y+X)
    
    
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    """
    function：
        残差连接模块
        
    Parameters:
        in_channels - 输入通道数
        out_channels - 输出通道数
        num_residuals - 残差块个数
        first_block - 是否是第一个残差块

    Returns:
        残差模块序列

    Modify:
        2020-12-24
    """
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)
    
    
def resnet18(output=10, in_channels=3):
    """
    function：
        18层残差网络
        
    Parameters:
        in_channels - 输入通道数
        out_channels - 输出通道数

    Returns:
        残差网络

    Modify:
        2020-12-24
    """
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), 
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
    net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, output))) 
    return net


def load_data_jay_lyrics():
    """
    function：
        加载周杰伦歌词数据
        
    Parameters:

    Returns:
        corpus_indices - 将训练样本转换为编码格式
        char_to_idx - 将字符与索引映射构建编码字典
        idx_to_char - 所有出现过的字符
        vocab_size - 编码字典的大小

    Modify:
        2020-12-16
    """
    with zipfile.ZipFile('data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[:10000]
    # 所有出现过的字符
    idx_to_char = list(set(corpus_chars))
    # 将字符与索引映射构建编码字典
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    # 编码字典的大小
    vocab_size = len(char_to_idx)
    # 将样本转换为编码格式
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size

   
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    """
    function：
        序列通过随机采样生成样本
        
    Parameters:
        corpus_indices - 编码后的原始序列
        batch_size - 批大小
        num_steps - 每一批中每个样本的大小
        device - 指定计算在GPU或者CPU上进行

    Returns:
        样本
        标签

    Modify:
        2020-12-16
    """
    # 减1是因为输出的索引x是相应输入的索引y加1
    # 这里是计算一共生成多少个样本
    num_examples = (len(corpus_indices) - 1) // num_steps
    # 循环多少次可以遍历所有样本
    epoch_size = num_examples // batch_size
    # 每一条样本的索引
    example_indices = list(range(num_examples))
    # 随机打乱样本索引
    random.shuffle(example_indices)
    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos+num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 构建每一个epoch内的样本
    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i+batch_size]
        X = [_data(j*num_steps) for j in batch_indices]
        Y = [_data(j*num_steps+1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)
        

def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    """
    function：
        序列通过相邻采样生成样本
        
    Parameters:
        corpus_indices - 编码后的原始序列
        batch_size - 批大小
        num_steps - 每一批中每个样本的大小
        device - 指定计算在GPU或者CPU上进行

    Returns:
        样本
        标签

    Modify:
        2020-12-16
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    # 原始数据长度
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].view(batch_size, batch_len)
    # 减1是因为输出的索引x是相应输入的索引y加1
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y
    
    
def to_onehot(X, size):
    """
    function：
        对样本进行独热编码
        
    Parameters:
        X - 待编码样本
        size - 字典大小

    Returns:
        编码结果

    Modify:
        2020-12-17
    """
    return F.one_hot(X.t(), size)


def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state, num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    """
    function：
        利用训练好的rnn模型根据已知输入进行推理
        
    Parameters:
        prefix - 已知的输入
        num_chars - 要预测的字符个数
        rnn - 训练好的模型
        params - 模型参数
        init_rnn_state - 初始化的隐藏状态
        num_hiddens - 隐藏神经元个数
        vocab_size - 字典大小
        device - 指定计算在GPU或者CPU上进行
        idx_to_char - 索引到字符之间的映射
        char_to_idx - 字符到索引之间的映射

    Returns:
        预测结果

    Modify:
        2020-12-17
    """
    # 初始化的隐藏状态
    state = init_rnn_state(1, num_hiddens, device)
    # 将输入的首字符传入到输出序列中
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix)-1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


def grad_clipping(params, theta, device):
    """
    function：
        梯度裁剪，使模型的梯度值的L2范数小于等于theta
        
    Parameters:
        params - 模型参数
        theta - 阈值
        rnn - 训练好的模型

    Returns:
        裁剪后的梯度值

    Modify:
        2020-12-17
    """
    # 存储参数梯度值的平方和开根号
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data**2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta/norm)
           
        
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx, is_random_iter, num_epochs, num_steps, lr, clipping_theta, 
                         batch_size, pred_period, pred_len, prefixes):
    """
    function：
        循环神经网络的训练以及预测过程
        
    Parameters:
        rnn - 要使用的rnn模型
        get_params - 初始化模型参数
        init_rnn_state - 初始化隐藏状态
        num_hiddens - 隐层神经元个数
        vocab_size - 字典大小
        device - 指定计算在GPU或者CPU上进行
        corpus_indices - 编码之后的样本
        idx_to_char - 索引到字符之间的映射
        char_to_idx - 字符到索引之间的映射
        is_random_iter - 是否采用随机采样
        num_epochs - 训练轮次
        num_steps - 每一批中每个样本的大小
        lr - 学习率
        clipping_theta - 梯度裁剪阈值
        batch_size - 批大小
        pred_period - 每多少次打印一下训练结果
        pred_len - 生成样本长度
        prefixes - 传入的生成引导内容

    Returns:

    Modify:
        2020-12-17
    """
    if is_random_iter:
        # 随机采样
        data_iter_fn = data_iter_random
    else:
        # 相邻采样
        data_iter_fn = data_iter_consecutive
    # 初始化模型参数
    params = get_params()
    # 交叉熵损失
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        if not is_random_iter:
            # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, device)
        # 损失之和、样本数、起始时间
        l_sum, n, start = 0.0, 0, time.time()
        # 生成训练样本及label
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            # 如使用随机采样，在每个小批量更新前初始化隐藏状态
            if is_random_iter:
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:
                # 否则需要使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                for s in state:
                    s.detach()
            # 独热编码
            inputs = to_onehot(X.long(), vocab_size)
            # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
            (outputs, state) = rnn(inputs, state, params)
            # 拼接之后形状为(num_steps * batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)
            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            # 使用交叉熵损失计算平均分类误差
            l = loss(outputs, y.long())
            # 梯度清0
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            # https://blog.csdn.net/qq_29695701/article/details/89965986
            l.backward(retain_graph=True)
            # 裁剪梯度
            grad_clipping(params, clipping_theta, device)
            # 因为误差已经取过均值，梯度不用再做平均
            sgd(params, lr, 1)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch+1, math.exp(l_sum/n), time.time()-start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state, num_hiddens, vocab_size, device, idx_to_char, char_to_idx))
                      

class RNNModel(nn.Module):
    """
    function：
        循环神经网络模型
        
    Parameters:
        rnn_layer - RNN层
        vocab_size - 字典大小
        inputs - 编码之后的样本
        state - 初始化隐藏状态

    Returns:
        output - 模型输出
        state - 隐藏状态

    Modify:
        2020-12-17
    """
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        # 若为双向循环网络需要×2
        self.hidden_size = rnn_layer.hidden_size*(2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None
    # inputs: (batch, seq_len)
    def forward(self, inputs, state):
        # 获取one-hot向量表示
        # X是个list
        X = to_onehot(inputs.long(), self.vocab_size)
        Y, self.state = self.rnn(X.float(), state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)
        # 它的输出形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state
    

def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char, char_to_idx):
    """
    function：
        使用循环神经网络模型推理
        
    Parameters:
        prefixe - 传入的生成引导内容
        num_chars - 生成样本长度
        model - 要使用的rnn模型
        vocab_size - 字典大小
        device - 指定计算在GPU或者CPU上进行
        idx_to_char - 索引到字符之间的映射
        char_to_idx - 字符到索引之间的映射

    Returns:
        生成内容

    Modify:
        2020-12-17
    """
    state = None
    # output会记录prefix加上输出
    # 将输入的首字符传入到输出序列中
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars+len(prefix)-1):
        # 将上一时间步的输出作为当前时间步的输入
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple):
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)
        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx, num_epochs, num_steps, lr, clipping_theta, 
                                  batch_size, pred_period, pred_len, prefixes):
    """
    function：
        PyTorch版循环神经网络的训练以及预测过程
        
    Parameters:
        model - 要使用的rnn模型
        num_hiddens - 隐层神经元个数
        vocab_size - 字典大小
        device - 指定计算在GPU或者CPU上进行
        corpus_indices - 编码之后的样本
        idx_to_char - 索引到字符之间的映射
        char_to_idx - 字符到索引之间的映射
        num_epochs - 训练轮次
        num_steps - 每一批中每个样本的大小
        lr - 学习率
        clipping_theta - 梯度裁剪阈值
        batch_size - 批大小
        pred_period - 每多少次打印一下训练结果
        pred_len - 生成样本长度
        prefixes - 传入的生成引导内容

    Returns:

    Modify:
        2020-12-17
    """
    # 交叉熵损失
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(num_epochs):
        # 损失之和、样本数、起始时间
        l_sum, n, start = 0.0, 0, time.time()
        # 相邻采样
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if state is not None:
                # 否则需要使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                if isinstance(state, tuple):
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()
            # output: 形状为(num_steps * batch_size, vocab_size)
            (output, state) = model(X, state)
            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            # 使用交叉熵损失计算平均分类误差
            l = loss(output, y.long())
            # 梯度清0
            optimizer.zero_grad()

            # https://blog.csdn.net/qq_29695701/article/details/89965986
            l.backward()
            # 裁剪梯度
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch+1, perplexity, time.time()-start))
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(prefix, pred_len, model, vocab_size, device, idx_to_char, char_to_idx))
                

def train_2d(trainer):
    """
    function：
        获取梯度下降过程中自变量的计算值
        
    Parameters:
        trainer - 参数更新方式

    Returns:
        results - 每一次更新的参数值

    Modify:
        2020-12-22
    """
    # s1和s2是自变量状态，本章后续几节会使用
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i+1, x1, x2))
    return results


def show_trace_2d(f, results):
    """
    function：
        两个自变量的梯度更新可视化
        
    Parameters:
        f - 目标函数
        results - 每一步梯度更新计算结果

    Returns:

    Modify:
        2020-12-22
    """
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    
def get_data_ch7():
    """
    function：
        读取飞机机翼噪声的数据集
        
    Parameters:

    Returns:
        前1500个样本的特征
        前1500个样本的标签

    Modify:
        2020-12-23
    """
    # 数据读取以\t分隔
    data = np.genfromtxt('data/airfoil_self_noise.dat', delimiter='\t')
    # 特征标准化处理
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    # 最后一列是label
    # 前1500个样本(每个样本5个特征)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), torch.tensor(data[:1500, -1], dtype=torch.float32)


def train_ch7(optimizer_fn, states, hyperparams, features, labels, batch_size=10, num_epochs=2):
    """
    function：
        训练模型并绘制损失曲线
        
    Parameters:
        optimizer_fn - 优化器
        states - 状态
        hyperparams - 用字典定义的超参
        features - 特征
        labels - 标签
        batch_size - 小批量大小
        num_epochs - 迭代次数

    Returns:

    Modify:
        2020-12-23
    """
    # 初始化模型
    net, loss = linreg, squared_loss
    w = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(features.shape[1], 1)), dtype=torch.float32), requires_grad=True)
    b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)
    def eval_loss():
        return loss(net(features, w, b), labels).mean().item()
    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features, labels), batch_size, shuffle=True)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            # 使用平均损失
            l = loss(net(X, w, b), y).mean()
            # 梯度清零
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            # 迭代模型参数
            optimizer_fn([w, b], states, hyperparams)
            # 每100个样本记录下当前训练误差
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time()-start))
    set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    
    
def train_pytorch_ch7(optimizer_fn, optimizer_hyperparams, features, labels, batch_size=10, num_epochs=2):
    """
    function：
        训练模型并绘制损失曲线
        
    Parameters:
        optimizer_fn - 优化器
        optimizer_hyperparams - 用字典定义的超参
        features - 特征
        labels - 标签
        batch_size - 小批量大小
        num_epochs - 迭代次数

    Returns:

    Modify:
        2020-12-23
    """
    # 初始化模型
    net = nn.Sequential(
        nn.Linear(features.shape[-1], 1)
    )
    loss = nn.MSELoss()
    optimizer = optimizer_fn(net.parameters(), **optimizer_hyperparams)
    def eval_loss():
        return loss(net(features).view(-1), labels).item() / 2
    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels), batch_size, shuffle=True
    )
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            # 除以2是为了和train_ch7保持一致, 因为squared_loss中除了2
            l = loss(net(X).view(-1), y) / 2
            # 梯度清零
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time()-start))
    set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    
    
class Benchmark():
    """
    function：
        计算程序执行时间
        
    Parameters:
        prefix - 要打印的前置字符串

    Returns:

    Modify:
        2020-12-24
    """
    def __init__(self, prefix=None):
        # 打印的前置字符串
        self.prefix = prefix + ' ' if prefix else ''
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, *args):
        print('%stime: %.4f sec' % (self.prefix, time.time()-self.start))
        
        
def show_images(imgs, num_rows, num_cols, scale=2):
    """
    function：
        绘图程序
        
    Parameters:
        imgs - 要绘制的所有图片
        num_rows - 行数
        num_cols - 列数
        scale - 缩放比

    Returns:
        axes - 绘制结果

    Modify:
        2020-12-24
    """
    figsize = (num_cols*scale, num_rows*scale)
    # 子图布局
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i*num_cols+j])
            # 不显示坐标轴
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes


def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    """
    function：
        训练程序
        
    Parameters:
        train_iter - 训练数据
        test_iter - 测试数据
        net - 网络
        loss - 损失函数
        optimizer - 优化器
        device - 指定计算在GPU或者CPU上进行
        num_epochs - 迭代次数

    Returns:

    Modify:
        2020-12-24
    """
    # 多卡并行
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = torch.nn.DataParallel(net, device_ids=[0, 1])
    net = net.cuda()
    print("training on", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device).long()
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1).long()==y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec' % (epoch+1, train_l_sum/batch_count, train_acc_sum/n, test_acc, time.time()-start))
        
        
def bbox_to_rect(bbox, color):
    """
    function：
        绘制边界框
        将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
        ((左上x, 左上y), 宽, 高)
        
    Parameters:
        bbox - 边界框左上角、右下角坐标
        color - 边界框边的颜色

    Returns:

    Modify:
        2020-12-25
    """
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1], 
        fill=False, edgecolor=color, linewidth=2
    )


def MultiBoxPrior(feature_map, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5]):
    """
    function：
        给图像添加锚框
        
    Parameters:
        feature_map: torch tensor, Shape: [N, C, H, W].
        sizes: List of sizes (0~1) of generated MultiBoxPriores. 
        ratios: List of aspect ratios (non-negative) of generated MultiBoxPriores. 

    Returns:
        anchors of shape (1, num_anchors, 4). 由于batch里每个都一样, 所以第一维为1

    Modify:
        2020-12-25
    """
    # 记录锚框面积和宽高比的组合方式
    pairs = []
    # 包含s1或r1的组合方式
    for r in ratios:
        pairs.append([sizes[0], math.sqrt(r)])
    for s in sizes[1:]:
        pairs.append([s, math.sqrt(ratios[0])])
    pairs = np.array(pairs)
    # size * sqrt(ration)
    ss1 = pairs[:, 0] * pairs[:, 1]
    # size / sqrt(ration)
    ss2 = pairs[:, 0] / pairs[:, 1]
    # 锚框左上角、右下角坐标计算
    # 除以2是因为像素点在锚框中心位置
    base_anchors = np.stack([-ss1, -ss2, ss1, ss2], axis=1) / 2
    # 目标图像的高、宽
    h, w = feature_map.shape[-2:]
    # x轴
    shifts_x = np.arange(0, w) / w
    # y轴
    shifts_y = np.arange(0, h) / h
    # 生成图布内所有坐标点
    shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
    # 拉成1维，位置相同的元素组成画布内的一个点的表示
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    shifts = np.stack((shift_x, shift_y, shift_x, shift_y), axis=1)
    # 每个像素点添加所有类型的锚框
    anchors = shifts.reshape((-1, 1, 4))+base_anchors.reshape((1, -1, 4))
    return torch.tensor(anchors, dtype=torch.float32).view(1, -1, 4)


def MultiBoxPrior_My(feature_map, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5]):
    """
    function：
        给图像添加锚框,按照自己推导的公式设计
        
    Parameters:
        feature_map: torch tensor, Shape: [N, C, H, W].
        sizes: List of sizes (0~1) of generated MultiBoxPriores. 
        ratios: List of aspect ratios (non-negative) of generated MultiBoxPriores. 

    Returns:
        anchors of shape (1, num_anchors, 4). 由于batch里每个都一样, 所以第一维为1

    Modify:
        2020-12-25
    """
    # 目标图像的高、宽
    h, w = feature_map.shape[-2:]
    # 记录锚框面积和宽高比的组合方式
    pairs = []
    # 包含s1或r1的组合方式
    for r in ratios:
        pairs.append([sizes[0], math.sqrt(r)])
    for s in sizes[1:]:
        pairs.append([s, math.sqrt(ratios[0])])
    pairs = np.array(pairs)
    # size * sqrt(ration)
    ss1 = pairs[:, 0] * math.sqrt(h*w) * pairs[:, 1] / w
    # size / sqrt(ration)
    ss2 = pairs[:, 0] * math.sqrt(h*w) / (pairs[:, 1] * h)
    # 锚框左上角、右下角坐标计算
    # 除以2是因为像素点在锚框中心位置
    base_anchors = np.stack([-ss1, -ss2, ss1, ss2], axis=1) / 2
    # x轴
    shifts_x = np.arange(0, w) / w
    # y轴
    shifts_y = np.arange(0, h) / h
    # 生成图布内所有坐标点
    shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
    # 拉成1维，位置相同的元素组成画布内的一个点的表示
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    shifts = np.stack((shift_x, shift_y, shift_x, shift_y), axis=1)
    # 每个像素点添加所有类型的锚框
    anchors = shifts.reshape((-1, 1, 4))+base_anchors.reshape((1, -1, 4))
    return torch.tensor(anchors, dtype=torch.float32).view(1, -1, 4)


def show_bboxes(axes, bboxes, labels=None, colors=None):
    """
    function：
        给图像添加锚框
        
    Parameters:
        axes: 图片坐标
        bboxes: 要添加的锚框
        labels: 锚框上标注的文字
        colors: 锚框颜色

    Returns:

    Modify:
        2020-12-28
    """
    # 获取锚框上的标签
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    labels = _make_list(labels)
    # 框的颜色
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().cpu().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color=='w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], 
                     va='center', ha='center', fontsize=6, color=text_color, 
                     bbox=dict(facecolor=color, lw=0))
    # d2l.plt.savefig('1.jpg')
    
    
# 参考https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py#L356
def compute_intersection(set_1, set_2):
    """
    function：
        计算anchor之间的交集
        
    Args:
        set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
        set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
        
    Returns:
        intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
        
    Modify:
        2020-12-28
    """
    # PyTorch auto-broadcasts singleton dimensions
    # 左上角取最大
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    # 右下角取最小
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def compute_jaccard(set_1, set_2):
    """
    function：
        计算anchor之间的Jaccard系数(IoU)
        
    Args:
        set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
        set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
        
    Returns:
        Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
        
    Modify:
        2020-12-28
    """
    # Find intersections
    intersection = compute_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


class PiKachuDetDataset(torch.utils.data.Dataset):
    """
    function：
        构建皮卡丘样本
        
    Args:
        data_dir: 图片路径
        part: 指定训练集还是测试集
        image_size: 图片大小
        
    Returns:
        sample: 构建好的样本
        
    Modify:
        2020-12-30
    """
    def __init__(self, data_dir, part, image_size=(256, 256)):
        assert part in ['train', 'val']
        # 图片尺寸
        self.image_size = image_size
        # 图片路径
        self.image_dir = os.path.join(data_dir, part, 'images')
        # 图片标签
        with open(os.path.join(data_dir, part, 'label.json')) as f:
            self.label = json.load(f)
        self.transform = torchvision.transforms.Compose([
            # 将 PIL 图片转换成位于[0.0, 1.0]的floatTensor, shape (C x H x W)
            torchvision.transforms.ToTensor()
        ])
    def __len__(self):
        # 样本个数
        return len(self.label)
    def __getitem__(self, index):
        # 构建样本
        image_path = str(index+1)+'.png'
        cls = self.label[image_path]['class']
        label = np.array([cls]+self.label[image_path]['loc'], dtype='float32')[None, :]
        PIL_img = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB').resize(self.image_size)
        img = self.transform(PIL_img)
        sample = {
            # shape: (1, 5) [class, xmin, ymin, xmax, ymax]
            'label': label, 
            # shape: (3, *image_size)
            'image': img
        }
        return sample
    
    
def load_data_pikachu(batch_size, edge_size=256, data_dir='data/pikachu'):
    """
    function：
        分别构建皮卡丘数据集的训练集样本和测试集样本
        
    Args:
        batch_size: 小批量大小
        edge_size: 输出图片的维度
        data_dir: 图片路径
        
    Returns:
        train_iter: 训练集样本
        val_iter: 测试集样本
        
    Modify:
        2020-12-30
    """
    # edge_size：输出图像的宽和高
    image_size = (edge_size, edge_size)
    train_dataset = PiKachuDetDataset(data_dir, 'train', image_size)
    val_dataset = PiKachuDetDataset(data_dir, 'val', image_size)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_iter, val_iter


def read_voc_images(root='data/VOCdevkit/VOC2012', is_train=True, max_num=None):
    """
    function：
        读取语义分割数据集集标签
        
    Args:
        root: 数据存储路径
        is_train: 是否为训练集
        max_num: 最大读取数据量
        
    Returns:
        features: 图片(PIL)
        labels: 标签(PIL)
        
    Modify:
        2021-01-11
    """
    txt_fname = '%s/ImageSets/Segmentation/%s' % (
        root, 'train.txt' if is_train else 'val.txt'
    )
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    if max_num is not None:
        images = images[:min(max_num, len(images))]
    features, labels = [None]*len(images), [None]*len(images)
    for i, fname in tqdm(enumerate(images)):
        features[i] = Image.open('%s/JPEGImages/%s.jpg' % (root, fname)).convert('RGB')
        labels[i] = Image.open('%s/SegmentationClass/%s.png' % (root, fname)).convert('RGB')
    return features, labels


def voc_label_indices(colormap, colormap2label):
    """
    function：
        将colormap(PIL)转换为colormap2label(uint8)
        
    Args:
        colormap: 图片颜色标签
        colormap2label: 像素级数值标签
        
    Returns:
        该图片对应的标签转换结果
        
    Modify:
        2021-01-11
    """
    colormap = np.array(colormap.convert('RGB')).astype('int32')
    idx = ((colormap[:, :, 0]*256+colormap[:, :, 1])*256+colormap[:, :, 2])
    return colormap2label[idx]


def voc_rand_crop(feature, label, height, width):
    """
    function：
        同时对图片和标签进行随机裁剪为固定大小
        
    Args:
        feature: 图片
        label: 标签
        height: 裁剪之后的高
        width: 裁剪之后的宽
        
    Returns:
        feature: 裁剪之后的图片
        label: 裁剪之后的标签
        
    Modify:
        2021-01-11
    """
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(
        feature, output_size=(height, width)
    )
    feature = torchvision.transforms.functional.crop(feature, i, j, h, w)
    label = torchvision.transforms.functional.crop(label, i, j, h, w)
    return feature, label


class VOCSegDataset(torch.utils.data.Dataset):
    """
    function：
        对VOC2012数据集做预处理
        
    Args:
        is_train: 是否为训练集
        crop_size: 裁剪之后样本的大小
        voc_dir: 图片存储路径
        colormap2label: 将图片标签按照像素转换
        max_num: 最大读取样本数
        
    Returns:
        feature: 转换之后的图片
        label: 转换之后的标签
        
    Modify:
        2021-01-11
    """
    def __init__(self, is_train, crop_size, voc_dir, colormap2label, max_num=None):
        # crop_size - 随机裁剪后的大小
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.tsf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize(mean=self.rgb_mean, 
                                            std=self.rgb_std)
        ])
        self.crop_size = crop_size #(h, w)
        features, labels = read_voc_images(root=voc_dir, 
                                          is_train=is_train, 
                                          max_num=max_num)
        self.features = self.filter(features)
        self.labels = self.filter(labels)
        self.colormap2label = colormap2label
        print('read ' + str(len(self.features)) + ' valid examples')
    def filter(self, imgs):
        return [img for img in imgs if (
            img.size[1] >= self.crop_size[0] and
            img.size[0] >= self.crop_size[1]
        )]
    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
        # float32, uint8
        return (self.tsf(feature), voc_label_indices(label, self.colormap2label))
    def __len__(self):
        return len(self.features)
    
    
def train_ch8(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    """
    function：
        训练程序
        
    Parameters:
        train_iter - 训练数据
        test_iter - 测试数据
        net - 网络
        loss - 损失函数
        optimizer - 优化器
        device - 指定计算在GPU或者CPU上进行
        num_epochs - 迭代次数

    Returns:

    Modify:
        2020-12-24
    """
    # 多卡并行
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = torch.nn.DataParallel(net, device_ids=[0, 1])
    net = net.cuda()
    print("training on", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device).long()
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1).long()==y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec' % (epoch+1, train_l_sum/batch_count, train_acc_sum/(n*153600.0), test_acc/153600.0, time.time()-start))
        
        
def read_csv_labels(fname):
    """
    function：
        从csv文件读取图片的标签
        
    Parameters:
        fname - 文件路径

    Returns:
        文件名到标签之间的映射

    Modify:
        2021-01-13
    """
    with open(fname, 'r') as f:
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))


def copyfile(filename, target_dir):
    """
    将文件复制到目标路径下
    """
    # 在路径不存在的情况下创建路径
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)
def reorg_train_valid(data_dir, labels, valid_ratio):
    """
    function：
        划分出训练集与验证集
        
    Parameters:
        data_dir - 文件路径
        labels - 图片标签
        valid_ratio - 训练集与验证集的比例

    Returns:
        验证集中每一类的样本数

    Modify:
        2021-01-13
    """
    # 训练集中样本数最少的类别包含的样本数
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每一类的样本数
    n_valid_per_label = max(1, math.floor(n*valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train', 'train')):
        # 为每个训练集样本匹配类别
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', 'train', train_file)
        # 以类别名为文件夹名称保存数据
        # 训练集与验证集合并
        copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train_valid', label))
        if label not in label_count or label_count[label]<n_valid_per_label:
            # 仅包含验证集
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            # 仅包含训练集
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train', label))
    return n_valid_per_label


def reorg_test(data_dir):
    """
    function：
        划分出测试集
        
    Parameters:
        data_dir - 文件路径

    Returns:

    Modify:
        2021-01-13
    """
    # 测试集数据整理，类别名为unknown
    for test_file in os.listdir(os.path.join(data_dir, 'test', 'test')):
        copyfile(os.path.join(data_dir, 'test', 'test', test_file), 
                os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))
     
    
def read_imdb(folder='train', data_root='data/aclImdb'):
    """
    function：
        读取imdb数据集
        
    Parameters:
        folder - 文件夹名称
        data_root - 数据根目录

    Returns:
        data - 影评，标签

    Modify:
        2021-01-18
    """
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_root, folder, label)
        # 读取路径下所有文件
        for file in tqdm(os.listdir(folder_name)):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label=='pos' else 0])
    random.shuffle(data)
    return data


def get_tokenized_imdb(data):
    """
    function：
        将句中单词小写，并按照空格对句子切分为单词
        
    Parameters:
        data - list of [string, label]

    Returns:
        切分为单词的数据集

    Modify:
        2021-01-18
    """
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]


def get_vocab_imdb(data):
    """
    function：
        获取影评数据集中词频高于5组成的词袋
        
    Parameters:
        data - 切分为单词的数据集

    Returns:
        影评词袋

    Modify:
        2021-01-18
    """
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return Vocab.Vocab(counter, min_freq=5)


def preprocess_imdb(data, vocab):
    """
    function：
        通过补齐或截断将数据集中的影评变为相同长度
        
    Parameters:
        data - 原始数据集
        vocab - 切分为单词的数据集

    Returns:
        features - 编码后的影评
        labels - 情感标签

    Modify:
        2021-01-18
    """
    # 将每条评论通过截断或者补0，使得长度变成500
    max_l = 500
    def pad(x):
        return x[:max_l] if len(x)>max_l else x+[0]*(max_l-len(x))
    tokenized_data = get_tokenized_imdb(data)
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels


def load_pretrained_embedding(words, pretrained_vocab):
    """
    function：
        从预训练好的vocab中提取出words对应的词向量
        
    Parameters:
        words - 训练文本的词袋
        pretrained_vocab - 预训练词向量

    Returns:
        embed - 嵌入层的初始化数值

    Modify:
        2021-01-18
    """
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0])
    oov_count = 0
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i,:] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 1
    # 没有索引的单词个数
    if oov_count > 0:
        print('There are %d oov words.' % oov_count)
    return embed


def predict_sentiment(net, vocab, sentence):
    """
    function：
        预测一个句子的情感
        
    Parameters:
        net - 训练好的模型
        vocab - 词表
        sentence - 词语的列表

    Returns:
        句子的情感

    Modify:
        2021-01-18
    """
    device = list(net.parameters())[0].device
    sentence = torch.tensor([vocab.stoi[word] for word in sentence], device=device)
    label = torch.argmax(net(sentence.view((1, -1))), dim=1)
    return 'positive' if label.item()==1 else 'negative'


if __name__ == '__main__':
    pass
    # the sample of the function