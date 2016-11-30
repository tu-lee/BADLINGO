import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

#input data

lorange= 1
hirange= 10
amplitude= np.random.uniform(-10,10)
t= 10
random.seed()
tau=np.random.uniform(lorange,hirange)


#tensors for input data

x_input= tf.placeholder(tf.float32, shape=(10,))# t=10
y_input= tf.placeholder(tf.float32, shape=(10,))

#use 10 neurons-- just one layer for now

weights_1= tf.Variable(tf.truncated_normal([10,10], stddev= .1))
bias_1= tf.Variable(.1)

#hidden output
hidden_output= tf.nn.sigmoid(tf.matmul(tf.reshape(x_input,[1,10]), weights_1) + bias_1)


weights_2 = tf.Variable(tf.truncated_normal([10,10], stddev=.1))
bias_2= tf.Variable(.1)

calculated_output = tf.nn.softmax(tf.matmul(hidden_output, weights_2) + bias_2)
#clipped_output= tf.clip_by_value(calculated_output, 1e-37, 1e+37)

with tf.name_scope("cost") as scope:
  cost = tf.reduce_mean((tf.square(y_input-calculated_output)))
  cost = tf.scalar_summary("cost", cost)


with tf.name_scope("train") as scope:
  train_step = tf.train.GradientDescentOptimizer(.0001).minimize(cost)


#histogrms
weights_1_hist = tf.histogram_summary("weights_1", weights_1)
bias_1_hist = tf.histogram_summary("bias_1", bias_1)
y_hist = tf.histogram_summary("y", calculated_output)

weights_2_hist = tf.histogram_summary("weights", weights_2)
bias_2_hist = tf.histogram_summary("biases", bias_2)





#session
sess = tf.InteractiveSession()

#merge all summaries
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/mnist_logs_4", sess.graph_def)

init = tf.initialize_all_variables()
sess.run(init)
steps = 1000

for i in range(steps):
    xs = np.arange(t)
    ys = amplitude * np.exp(-xs / tau)

    feed = {x_input: xs, y_input: ys}
    sess.run(train_step, feed_dict=feed)

    error = tf.reduce_sum(tf.abs(clipped_output - y_input))
    print("After %d iteration:" % i)
    print("W: %s" % sess.run(weights_2))
    print("b: %s" % sess.run(bias_2))
    print('Total Error: ', error.eval(feed_dict={x_input: xs, y_input: ys}))
