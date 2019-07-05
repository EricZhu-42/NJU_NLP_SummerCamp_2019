import tensorflow as tf

INPUT_NODE = 784 #输入层节点数，此处等于图片像素
OUTPUT_NODE = 10 #输出层结点数，此处等于类别数目0-9

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512

REGULARIZATION_RATE = 1e-4 #正则化项损失系数

def inference(input_tensor, train, regularizer, avg_class=None):
    #第一层，卷积层，核尺寸5*5，深度为32，不使用全0填充，步长为1.
    #输入大小32*32*1，输出大小28*28*32
    with tf.variable_scope('layer1-conv1', reuse=tf.AUTO_REUSE):
        weights = tf.get_variable(
            'weights', [CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(
            'biases', [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        layer = tf.nn.conv2d(input_tensor,weights,strides=[1,1,1,1],padding='SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(layer, biases))

    #第二层，池化层，采用2*2最大池，长宽步长均为2.
    #输入大小28*28*32，输出大小14*14*32
    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    #第三层，卷积层，核尺寸5*5，深度为64，不使用全0填充，步长为1.
    #输入大小14*14*32，输出大小10*10*64
    with tf.variable_scope('layer3-conv2', reuse=tf.AUTO_REUSE):
        weights = tf.get_variable(
            'weights', [CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(
            'biases', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        layer = tf.nn.conv2d(pool1 ,weights,strides=[1,1,1,1],padding='VALID')
        conv2 = tf.nn.relu(tf.nn.bias_add(layer, biases))

    #第四层，池化层，采用2*2最大池，长宽步长均为2.
    #输入大小10*10*64，输出大小5*5*64
    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    #第五层，全连接层, 共512个节点，训练过程中加入50%的dropout
    with tf.variable_scope('layer5-fc1'):
        pool_shape = pool2.get_shape().as_list()
        #将5*5*16的矩阵拉直成向量，其中pool_shape[0]表示batch数据个数
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2,[pool_shape[0],nodes])

        weights = tf.get_variable(
            'weights',[nodes, FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(REGULARIZATION_RATE)(weights))
        biases = tf.get_variable(
            'biases', [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, weights) + biases)
        if train:
            fc1 = tf.nn.dropout(fc1, rate=0.5)

    #第六层，全连接层, 共10个节点
    with tf.variable_scope('layer6-fc2'):
        weights = tf.get_variable(
            'weights',[FC_SIZE, NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(REGULARIZATION_RATE)(weights))
        biases = tf.get_variable(
            'biases', [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        fc2 = tf.matmul(fc1, weights) + biases

    return fc2

