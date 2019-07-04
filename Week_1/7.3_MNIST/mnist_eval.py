import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,(None,mnist_inference.INPUT_NODE),name='x-input')
        y_ = tf.placeholder(tf.float32,(None,mnist_inference.OUTPUT_NODE),name='y-input')
        validate_feed = {x:mnist.validation.images, y_:mnist.validation.labels}

        y = mnist_inference.inference(x)

        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)

                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy,feed_dict = validate_feed)
                print("After {:s} training steps, validation accuracy = {:g}".format(global_step,accuracy_score))
            else:
                print("No checkpoint file found.")
                return

def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    evaluate(mnist)

if __name__ == "__main__":
    tf.app.run()