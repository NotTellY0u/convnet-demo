import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from numpy.random.mtrand import randint
from Demo import mnist_data

mnist = mnist_data.read_data_sets('data', one_hot=True)
sess = tf.InteractiveSession()
new_saver = tf.train.import_meta_graph('softmax_mnist.meta')
new_saver.restore(sess, 'softmax_mnist')
tf.get_default_graph()
tf.get_default_graph().as_graph_def()
x = sess.graph.get_tensor_by_name("Reshape:0")
y_conv = sess.graph.get_tensor_by_name("output:0")
# image_b = mnist.test.images[100]
# image_b = np.reshape(image_b, [-1, 784])
image_b = randint(0, mnist.test.images.shape[0])
image_b = mnist.test.images[image_b]
# image_b = Image.open("13.png")
image_b = np.reshape(image_b, [-1, 784])
result = sess.run(y_conv, feed_dict={x:image_b})
print(result)
print(sess.run(tf.argmax(result, 1)))
plt.imshow(image_b.reshape([28, 28]), cmap='Greys')
plt.show()