import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

BATCH_SIZE = 100 #Batch大小
TRAINING_STEPS = 10000 #训练轮数
LEARNING_RATE_BASE = 0.8 # 基础学习率
LEARNING_RATE_DECAY = 0.999 #学习率衰减率
MOVING_AVERAGE_DECAY = 0.999 #滑动平均衰减率

MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model')
MODEL_NAME = "model.ckpt"

def train(mnist):
    #定义输入变量
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32,(None,mnist_inference.INPUT_NODE),name='x-input')
        y_ = tf.placeholder(tf.float32,(None,mnist_inference.OUTPUT_NODE),name='y-input')

    with tf.name_scope('init'):
        #定义存储训练轮数的变量
        global_step = tf.Variable(0,trainable=False)
        #初始化滑动平均类
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

    with tf.name_scope('calc'):
        #计算不使用滑动平均的神经网络向前传播结果
        y = mnist_inference.inference(x)
        #在神经网络参数变量上使用滑动平均
        variables_average_op = variable_averages.apply(tf.trainable_variables())
        #计算使用滑动平均后的神经网络向前传播结果
        y_average = mnist_inference.inference(x,variable_averages)

    with tf.name_scope('loss_function'):
        #计算交叉熵
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
        #计算交叉熵平均值
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        #计算模型正则化损失
        regularization = tf.add_n(tf.get_collection('losses'))
        #计算总损失
        loss = cross_entropy_mean + regularization

    with tf.name_scope('train_step'):
        #设置指数衰减学习率
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,global_step,mnist.train.num_examples / BATCH_SIZE,LEARNING_RATE_DECAY)
        #设置训练步骤，采用梯度下降法最小化损失函数
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
        #组合训练步骤与滑动平均过程
        train_op = tf.group(
            train_step, #训练步骤
            variables_average_op #滑动平均值修改步骤
            )

    with tf.name_scope('input_reshape'):
        #显示输入图像
        image_shaped_input = tf.reshape(x,[-1,28,28,1])

    #保存训练过程信息
    tf.summary.scalar('loss', loss)
    tf.summary.image('input', image_shaped_input, 10)
    tf.summary.histogram('inference_output',y)
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(".\log",sess.graph)
        tf.global_variables_initializer().run()
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}

        for i in range(TRAINING_STEPS+1):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            summary, _, loss_value, step = sess.run([merged, train_step, loss, global_step], feed_dict = {x:xs, y_:ys})
            summary_writer.add_summary(summary,i)
            if i % 1000 ==0:
                print("After {:d} training steps, loss on training batch is {:g}".format(step,loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH,MODEL_NAME), global_step=global_step)

        summary_writer.close()

def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()
