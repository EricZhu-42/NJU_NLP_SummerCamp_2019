# 基于神经网络与MNIST数据集的分类任务

> 时间：2019-07-04

## 文件结构

|   名称   |  描述    |
| ---- | ---- |
|   MNIST_data\   |  数据集本身    |
|    log\  |   训练过程中的日志文件   |
|   model\   |  保存的训练完成的模型文件    |
|   data\   |  关于记录编写的其他文件    |
|   mnist_inference.py   |  模型的预测部分（前向传播）  |
|   mnist_train.py   |  模型的训练部分（反向传播）  |
|   mnist_eval.py   |  模型的评估器   |

## 模型结构细节

![Structure](data\structure.png)

### a. 预测部分

1. 输入层：784个节点（等于输入图片像素）
2. 隐藏层：共两层
   1. 该层共250个节点，采用RELU作为激活函数，L2正则避免过拟合，滑动平均模型优化。
   2. 该层共250个节点，采用RELU作为激活函数，L2正则避免过拟合，滑动平均模型优化。
3. 输出层：10个节点 （等于输出类别数目） ，使用L2正则避免过拟合，滑动平均模型优化。

### b.训练部分

1. 采用批量梯度下降法进行优化
2. 采用交叉熵平均值与正则化损失的和为损失函数
	$loss = cross\_entropy + regularizaion$ 
3. 采用指数衰减法调整学习率
4. 对训练完成的模型进行保存，便于后续评估调用

### c. 评估部分

1. 调用前阶段保存的模型，基于MNIST的测试集进行评估。

## 测试结果与反思

### 测试结果

对于4层神经网络（一层输入，两层隐含，一层输出），取Batch大小为100，训练次数为10k，该模型在测试集上可以达到最高98.34%的正确率。

![result](data\result.png)

### 结果反思

以下纪录在模型训练与调整中遇到的问题

1. 引入激活函数RELU后，对模型的正确率有了极大的提升（92%-->97%），引入L2正则化，滑动平均模型与指数衰减法后，进一步提高了模型的正确率（97%-->98%）。
2. 网络结构与神经元个数对正确率的影响：
   1. 对于3层神经网络（一层输入，一层隐含，一层输出），将隐含层节点个数由500提升至800后，正确率获得了显著的提升（97.2%-->98.0%）
   2. 对于4层神经网络（784+250+250+10），正确率略高于3层结构（784+800+10），深层神经网络可以更好地提取数据中的特征。

3. 在训练过程中，出现了损失函数突然增大的现象，需要解决 //TODO