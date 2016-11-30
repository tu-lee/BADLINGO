import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from bnf import *

#hyperparameters
Batch_size= 100
max_iteration= 50
learning_rate=1000.
filt_1= [2,2,1]
num_fc_1 = 10
dropout = 0.5
num_classes = 2

#training data
lorange= 1
hirange= 25
amplitude= np.random.uniform(-10,10 )
t= 100
random.seed()
tau=np.random.uniform(lorange,hirange)
def generate_data(randomsignal):
    X= np.arange(t)
    Y= amplitude*np.exp(-X/tau)
    return X, Y


X_node= tf.placeholder(tf.float32, shape= [None, 100])
Y_= tf.placeholder(tf.int32, shape= [None, 2])
Y_class= tf.argmax(Y_, dimension=1)
bn_train = tf.placeholder(tf.bool)
keep_prob = tf.placeholder('float', name = 'dropout_keep_prob')

def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name = name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name = name)


def conv2d(X, W):
  return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def rearrange(X_node):

   X_node = tf.reshape(X_node, [-1,10,10,1]) # height=10, width=10, channel=1
   return X_node

with tf.name_scope("Conv1") as scope:
  W_conv1 = weight_variable([filt_1[1], 1, 1, filt_1[0]], 'Conv_Layer_1')
  b_conv1 = bias_variable([filt_1[0]], 'bias_for_Conv_Layer_1')
  a_conv1 = conv2d(X_node, W_conv1) + b_conv1
  h_conv1 = tf.nn.relu(a_conv1)


with tf.name_scope('max_pool1') as scope:
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, filt_1[2], 1, 1],
                        strides=[1, filt_1[2], 1, 1], padding='VALID')

    width_pool1 = int(np.floor((10-filt_1[2])/filt_1[2]))+1
    size1 = tf.shape(h_pool1)

with tf.name_scope('Batch_norm1') as scope:
    a_bn1 = batch_norm(h_pool1,filt_1[0],bn_train,'bn')
    h_bn1 = tf.nn.relu(a_bn1)

with tf.name_scope("Fully_Connected1") as scope:
    W_fc1 = weight_variable([width_pool1 * filt_1[0], num_fc_1], 'Fully_Connected_layer_1')
    b_fc1 = bias_variable([num_fc_1], 'bias_for_Fully_Connected_Layer_1')
    h_flat = tf.reshape(h_bn1, [-1, width_pool1 * filt_1[0]])
    h_flat = tf.nn.dropout(h_flat, keep_prob)
    h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)



with tf.name_scope("Output_layer") as scope:
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  W_fc2 = tf.Variable(tf.truncated_normal([num_fc_1, num_classes], stddev=0.1),name = 'W_fc2')
  b_fc2 = tf.Variable(tf.constant(0.1, shape=[num_classes]),name = 'b_fc2')
  h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  size3 = tf.shape(h_fc2)


with tf.name_scope("SoftMax") as scope:
  error = tf.nn.sparse_softmax_cross_entropy_with_logits(h_fc2,Y_)
  cost = tf.reduce_sum(error) / Batch_size
  #error_summ = tf.scalar_summary("cross entropy_loss", cost)


with tf.name_scope("train") as scope:
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.name_scope("Evaluating") as scope:
    correct_prediction = tf.equal(tf.argmax(h_fc2, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for i in range(1000):

    X, Y = generate_data(Batch_size)


    #sess.run(train_step, feed_dict={X_node: X, Y_: Y})
print('Total Error: ', error.eval(feed_dict={X_node: X, Y_: Y}))


