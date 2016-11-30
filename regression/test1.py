import tensorflow as tf
import numpy as np
import random

lorange= 1
hirange= 10
amplitude= np.random.uniform(-10,10)
t= 10
random.seed()
tau=np.random.uniform(lorange,hirange)


x_node = tf.placeholder(tf.float32, (10,))
y_node = tf.placeholder(tf.float32, (10,))

W = tf.Variable(tf.truncated_normal([10,10], stddev= .1))
b = tf.Variable(.1)

y = tf.matmul(tf.reshape(x_node,[1,10]), W) + b


cost = tf.reduce_mean(tf.square(y_node-y))

train_step = tf.train.GradientDescentOptimizer(0.000001).minimize(cost)
error = tf.reduce_sum(tf.abs(y - y_node))

#session
sess = tf.InteractiveSession()
init = tf.initialize_all_variables()
sess.run(init)
steps = 10000

for i in range(steps):
    xs = np.arange(t)
    ys = amplitude * np.exp(-xs / tau)

    feed = {x_node: xs, y_node: ys}
    sess.run(train_step, feed_dict=feed)


    print("After %d iteration:" % i)
    print("W: %s" % sess.run(W))
    print("b: %s" % sess.run(b))
    print('Total Error: ', error.eval(feed_dict={x_node: xs, y_node: ys}))
