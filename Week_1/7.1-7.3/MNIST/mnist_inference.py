import tensorflow as tf

INPUT_NODE = 784 #输入层节点数，此处等于图片像素
OUTPUT_NODE = 10 #输出层结点数，此处等于类别数目0-9

LAYER1_NODE = 250 #隐藏层1节点数
LAYER2_NODE = 250 #隐藏层2节点数
REGULARIZATION_RATE = 1e-4 #正则化项损失系数

def inference(input_tensor, avg_class=None):

    #隐藏层1，采用RELU为激活函数，使用L2正则避免过拟合，使用滑动平均模型优化
    with tf.variable_scope('layer1', reuse=tf.AUTO_REUSE):
        weights = tf.get_variable("weights",[INPUT_NODE, LAYER1_NODE],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(weights))
        biases = tf.get_variable("biases",[LAYER1_NODE],
                                initializer=tf.constant_initializer(0.0))
        if avg_class==None:
            layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
        else:
            layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights)) + avg_class.average(biases))

    #隐藏层2，采用RELU为激活函数，使用L2正则避免过拟合，使用滑动平均模型优化
    with tf.variable_scope('layer2', reuse=tf.AUTO_REUSE):
        weights = tf.get_variable("weights",[LAYER1_NODE, LAYER2_NODE],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(weights))
        biases = tf.get_variable("biases",[LAYER2_NODE],
                                initializer=tf.constant_initializer(0.0))
        if avg_class==None:
            layer2 = tf.nn.relu(tf.matmul(layer1,weights)+biases)
        else:
            layer2 = tf.nn.relu(tf.matmul(layer1,avg_class.average(weights)) + avg_class.average(biases))

    #输出层，使用L2正则避免过拟合，使用滑动平均模型优化
    with tf.variable_scope('layer3', reuse=tf.AUTO_REUSE):
        weights = tf.get_variable("weights",[LAYER2_NODE, OUTPUT_NODE],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(weights))
        biases = tf.get_variable("biases",[OUTPUT_NODE],
                                initializer=tf.constant_initializer(0.0))
        if avg_class==None:
            layer3 = tf.matmul(layer2,weights) + biases
        else:
            layer3 = tf.matmul(layer2,avg_class.average(weights)) + avg_class.average(biases)

        return layer3