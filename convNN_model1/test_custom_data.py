import numpy as np
import tensorflow as tf
import random

lorange= 1
hirange= 10
amplitude= np.random.uniform(-10,10)
t= 10
random.seed()
tau=np.random.uniform(lorange,hirange)
xs = np.arange(t)
ys = amplitude * np.exp(-xs / tau)


x = tf.placeholder(tf.float32, (10,))
y_ = tf.placeholder(tf.float32, (10,))

def weight_variable(shape):
    initial = tf.truncated_normal([10,10], stddev= .1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding= 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,1,1,1], strides=[1,1,1,1], padding= 'SAME')

W_conv1 = weight_variable([5,5])
b_conv1 = bias_variable([5])

x_node= tf.reshape(x,[1,10])

h_conv1= tf.nn.relu(conv2d(x, W_conv1) + b)


