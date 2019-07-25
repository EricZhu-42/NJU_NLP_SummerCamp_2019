# Image Caption

>  时间：2019-07-22   
>  本项目是对 [*Show and Tell: A Neural Image Caption Generator*](https://arxiv.org/abs/1411.4555) 论文的复现
>  以下代码参考了 [此代码库](https://github.com/maz0318/nlpSummerCamp2019/tree/master/week4) 与 [此代码库](https://github.com/amundv/kth-sml-project)

## 文件结构

| 名称           | 描述                        |
| -------------- | --------------------------- |
| process.py     | COCO数据集的预处理脚本      |
| data_loader.py | COCO数据集的Dataset定义脚本 |
| model.py       | 模型的定义脚本              |
| train_nic.py   | 模型的训练脚本              |

## 数据集

本项目采用的数据集为 [Common Objects in Context](http://cocodataset.org/), 具体信息如下：

- Training: 2014 Contest Train images [83K images/13GB]
- Validation: 2014 Contest Val images [41K images/6GB]
- Test: 2014 Contest Test images [41K images/6GB]

## 运行环境

> 使用的运行环境如下：
>
> Pytorch ==
> Torchvision == 
>
> 训练机配置如下：
>
> PASS

## 模型结构

NIC结构由两个模型构成：Encoder与Decoder. 

- Encoder：卷积神经网络，采用经过在 `ImageNet` 上预训练的 `ResNet152` 模型 (`torchvision.models.resnet152`) 并进行微调（*fine tuning*），目的是提取图片特征，创建对图片特征进行语义描述的定长向量（Feature Vector）
- Decoder：循环神经网络，采用LSTM/GRU作为结构单元，目的是在 Feature Vector 的基础上生成对图片的自然语言描述文本（Caption）

