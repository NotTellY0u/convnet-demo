from random import randint

import tensorflow as tf
from Demo import mnist_data
import numpy as np

logs_path = 'log_simple_stats_softmax'
batch_size = 10000
learning_rate = 0.5
training_epochs = 1000

mnist = mnist_data.read_data_sets('data', one_hot=True)
X = tf.placeholder(tf.float32, [None, 28, 28, 1],name="input")
Y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
XX = tf.reshape(X, [-1, 784])
b = tf.Variable(tf.zeros([10]))
evidence = tf.matmul(XX, W) + b
Y = tf.nn.softmax(evidence,name="output")
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0
train_step = tf.train.GradientDescentOptimizer(0.005).\
                              minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(Y, 1),\
                              tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,\
                                  tf.float32))
tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(logs_path, \
                                graph=tf.get_default_graph())

    for epoch in range(training_epochs):
        batch_count = int(mnist.train.num_examples / batch_size)
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, summary = sess.run([train_step, summary_op],\
                                        feed_dict={XX: batch_x,\
                                        Y_: batch_y})
        writer.add_summary(summary,\
                        epoch * batch_count + i)
        print("Epoch: ", epoch)
    print("Accuracy: ", accuracy.eval\
                            (feed_dict={XX: mnist.test.images,\
                            Y_: mnist.test.labels}))
    print("done")
    num = randint(0, mnist.test.images.shape[0])
    img = mnist.test.images[num]
    classification = sess.run(tf.argmax(Y, 1), feed_dict={XX: [img]})
    print('Neural Network predicted', classification[0])
    print('Real label is:', np.argmax(mnist.test.labels[num]))
    saver = tf.train.Saver()

    save_path = saver.save(sess, "softmax_mnist")
    print("Model saved to %s" % save_path)