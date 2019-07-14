# 基于CNN的CIFAR-10数据集分类任务

> 时间：2019-07-06  
以下代码参考了Tensorflow的[Advanced CNN教程](https://www.tensorflow.org/tutorials/images/deep_cnn)与[示例代码](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10/)  
## 图像输入部分

1. 数据读取：用CPU进行所有的图像读取与预处理工作，用GPU进行模型的训练工作，提高模型训练的效率。在训练开始时，预处理20000张处理过的CIFAR图像填充到随机化处理队列中，避免图像I/O过程影响模型的训练速度。  

2. 图像增强（训练过程中）：对原始图像进行随机切割，翻转，调整（随机失真），增大训练样本的数据量。

   1. 切割：略小于原始图像，增加训练数据量并减小计算量

   2. 翻转：对图像随机进行左右翻转

   3. 亮度调整：对图像随机进行亮度调整（在一定范围内）

   4. 对比度调整：对比度增大阈值大于减小阈值（高对比度常常有助于识别）

3. 图像增强（测试过程中）：

   1. 切割：从原图像中心进行切割，防止影响图像主体
   2. 标准化：对原图像的RGB值进行线性标准化，使模型对图像的动态范围变化不敏感。

## 模型预测部分

该网络在AlexNet的基础上进行了一定修改，其模型结构如下。

| 层名称         | 说明                                                         |
| -------------- | ------------------------------------------------------------ |
| conv1          | 采用5x5卷积核，步长为1，全0填充，过滤器深度为64，激活函数为ReLU |
| pool1          | 采用3x3最大池，步长为2*                                      |
| norm1          | LRN层，对同一层响应较小的神经元进行抑制                      |
| conv2          | 采用5x5卷积核，步长为1，全0填充，过滤器深度为64，激活函数为ReLU |
| norm2          |  LRN层，对同一层响应较小的神经元进行抑制                                                          |
| pool2          | 采用3x3最大池，步长为2                                       |
| local3         | 含有384个节点的全连接层，激活函数为ReLU                      |
| local4         | 含有192个节点的全连接层，激活函数为ReLU                      |
| softmax_linear | 生成最终结果的softmax层                                      |

*：重叠池化（Overlapping Pooling），步长小于池化范围，可以抽取更强的特征表达，但增大了计算量。

## 模型训练部分

模型的目标函数是求交叉熵损失（Cross_entropy）和所有权重衰减项（L2）的和。使用标准的梯度下降法对参数进行优化，采用滑动平均值调整参数，用指数衰减法调整学习率。

## 模型调整
> 训练机配置如下：
>
> ​	CPU : I7-8700 (6C12T)   
> ​	GPU: RTX2060 (6G)   
> ​	Tensorflow-gpu = 1.14.0  
> ​	CUDA = v10.0   
> ​	cuDNN = v7.3.1  
>
> 原始参数如下：
>
    > ```python
    > Batch_size = 128
    > Steps = 20000 
    > Moving_Average_Decay = 0.9999
    > Num_Epochs_per_decay = 350
    > Learning_Rate_Decay_Factor = 0.1
    > nitial_Learning_Rate = 0.1
    > ```

| 改动描述                                         | 训练速度                     | 测试集正确率 |
| ------------------------------------------------ | ---------------------------- | ------------ |
| 原始参数                                         | 9600 ex./s, 0.013 sec/batch  | 89.9%        |
| Batch_size = 256 (增大一倍)                      | 11000 ex./s, 0.023 sec/batch | 92.7%        |
| 不使用滑动平均                                   | 9800 ex./s, 0.013sec/batch   | 87.6%        |
| 删去两个LRN层                                    | 13000 ex./s, 0.010sec/batch  | 90.0%        |
| 在local4与softmax_linear间加入84神经元的全连接层 | 9200 ex./s, 0.014sec/batch   | 89.4%        |
| Batch_size = 256, 训练次数= 50k                  | 11000 ex./s, 0.023sec/batch  | 96.1%        |

### 调整结果

1. 增大Batch_size有助于加快模型收敛，提高模型的精度，但会增大内存负担，使得网络的训练过程需要经过更多个epoches.
2. 滑动平均模型在一定程度上优化了模型的表现；学习率的指数衰减使得较多的训练次数产生效果。
3. LRN层果然好像没有什么用...