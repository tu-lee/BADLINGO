import tensorflow as tf
import numpy as np
import random

#x and y creation
lorange= 1
hirange= 10
amplitude= np.random.uniform(-10,10)
t= 10
random.seed()
tau=np.random.uniform(lorange,hirange)

#collect/prepare data
data_point_size=1000
batch_size=1
actual_b= 10
learning_rate=0.00001
log_file="/tmp/mnist_batch_1"

#model regression
x_node = tf.placeholder(tf.float32, (10,))
y_node = tf.placeholder(tf.float32, (10,))

W = tf.Variable(tf.zeros([10,10], name= "w"))
b = tf.Variable(tf.zeros([10]), name= "b")

y = tf.nn.softmax(tf.matmul(tf.reshape(x_node,[1,10]), W) + b)

##ADD SUMMARY

W_hist = tf.histogram_summary("weights", W)
b_hist = tf.histogram_summary("biases", b)
y_hist = tf.histogram_summary("y", y)

# Cost function sum((y_-y)**2)
with tf.name_scope("cost") as scope:
  cost = tf.reduce_mean(tf.square(y_node-y))
  cost_sum = tf.scalar_summary("cost", cost)

# Training using Gradient Descent to minimize cost
with tf.name_scope("train") as scope:
  train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)


sess = tf.InteractiveSession()

# Merge all the summaries and write them out to logfile
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/mnist_logs_4", sess.graph_def)
error = tf.reduce_sum(tf.abs(y - y_node))
init = tf.initialize_all_variables()
sess.run(init)

steps = 1000

for i in range(steps):
    xs = np.arange(t)
    ys = amplitude * np.exp(-xs / tau)

    feed = {x_node: xs, y_node: ys}
    sess.run(train_step, feed_dict=feed)
    print("After %d iteration:" % i)
    print("W: %s" % sess.run(W))
    print("b: %s" % sess.run(b))
    print('Total Error: ', error.eval(feed_dict={x_node: xs, y_node: ys}))
    # Record summary data, and the accuracy every 10 steps
    if i % 10 == 0:
      result = sess.run(merged, feed_dict=feed)
      writer.add_summary(result, i)