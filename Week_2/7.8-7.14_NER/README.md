# 基于BiLSTM+CRF的中文NER任务

> 时间：2019-07-11  
> 以下部分代码参考了[这篇文章](https://www.jianshu.com/p/6668b965583e)  
> [此处](https://github.com/EricZhu-42/NJU_NLP_SummerCamp_2019/tree/master/Week_2/7.8-7.14_NER)为本文GitHub代码库链接

## 文件结构

|   名称   |  描述    |
| ---- | ---- |
|   data\   | 所用的数据集（来源：[链接](https://github.com/zjy-ucas/ChineseNER)） |
|   model\   |  保存的训练完成的模型文件    |
| img\ | 文档中展示的部分图片 |
| NER_data.py | 数据预处理脚本 |
| NER_model.py | 模型的构造脚本 |
| NER_train.py | 模型的训练脚本 |
| NER_eval.py | 模型表现的评估脚本 |
## 测试环境
>使用的测试环境如下：
>
>Tensorflow-gpu == 1.14.0  
>Keras == 2.2.4  
>Numpy = 1.16.4    
>
>训练机配置如下：
>
>CPU : I7-8700 (6C12T)   
>​GPU: RTX2060 (6G)    
>​CUDA == v10.0   
>​cuDNN == v7.3.1  

## 处理流程

1. 数据预处理：读取训练数据，构造词典库。
2. 模型构造：初始化网络模型。
3. 模型训练：用训练集（共20864个句子）进行训练
4. 模型评估：用测试集（共4636个句子）对模型的Precision 、Recall与F1-score进行评估

## 模型结构

模型结构参考了[此项目](https://github.com/Determined22/zh-NER-TF)的模型结构，具体结构如下（图片来自[此链接](https://github.com/Determined22/zh-NER-TF/tree/master/pics)）

![图片](img/pic1.png)

#### 1. look-up layer

将每个字符的表示形式由 One-hot 向量转化为 Character Embedding. （此处的Embedding为随机初始化得到，效果较差，可考虑采用训练完成的Embedding进行初始化，可较大幅度地提升模型的表现）

#### 2. BiLSTM layer

双向的LSTM层，将句子与词语中的特征进行抽取。BiLSTM相较单向的LSTM而言可以更好地捕捉双向的语义依赖。

#### 3. CRF layer

为句子中的每个字符生成标签。CRF相较于Softmax可更好的利用字符在语句层面上包含的信息，从而提高分类的准确性。

## 模型调整

#### Batch_size = 16, Epoches = 3

| 类别  | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| O     | 98.36%    | 98.91% | 0.9863   |
| B-PER | 61.76%    | 70.00% | 0.6562   |
| I-PER | 59.65%    | 73.91% | 0.6602   |
| B-LOC | 76.00%    | 71.25% | 0.7355   |
| I-LOC | 68.03%    | 66.94% | 0.6748   |
| B-ORG | 67.14%    | 74.60% | 0.7068   |
| I-ORG | 77.94%    | 49.53% | 0.6057   |
| Mean  | 96.97%    | 97.08% | 0.9697   |

#### Batch_size = 32, Epoches = 4

| 类别  | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| O     | 98.58%    | 98.68% | 0.9863   |
| B-PER | 57.14%    | 70.59% | 0.6316   |
| I-PER | 53.12%    | 68.00% | 0.5965   |
| B-LOC | 68.75%    | 57.89% | 0.6286   |
| I-LOC | 69.23%    | 49.09% | 0.5745   |
| B-ORG | 47.62%    | 47.62% | 0.4762   |
| I-ORG | 67.14%    | 74.60% | 0.7068   |
| Mean  | 96.88%    | 96.87% | 0.9684   |

## 问题反思

1. 数据量较小，仅有约2万个句子作为训练集，导致模型表现不佳。
2. 采用了宽度为100*2的BiLSTM进行特征抽取，为此会将所有句子统一padding为长度200后进行输入。因此，调整LSTM层的宽度没有对模型的表现产生明显的影响。
