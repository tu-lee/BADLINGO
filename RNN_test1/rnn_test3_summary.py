import numpy as np
import random
import math
from mlxtend.preprocessing import one_hot
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

#parameters
num_hidden = 12
batch_size = 1
no_of_batches = 1
epoch = 1000
num_classes=10
lorange= 1
resolution= 500
hirange= 1000
amplitude= np.random.uniform(-10,10)
t= np.linspace(0,10, num =10)
random.seed()

tau= np.random.uniform(lorange,hirange)
X= np.array([amplitude * np.exp(-t / tau)])#.reshape((1,10,1))
print 'X', X

#------------Sorting out the input
train_input = X
train_input = tf.reshape(train_input,[batch_size,len(t),1])
print 'ti', train_input


#------------sorting out the output
train_output= [int(math.ceil(tau/resolution))]
train_output= one_hot(train_output, num_labels=10)
print 'onehot', train_output
test_input = train_output
test_output = train_output

#tf graph input

data = tf.placeholder(tf.float32, shape= [batch_size,len(t),1], name= "Data")
target = tf.placeholder(tf.float32, shape = [batch_size, num_classes], name= "Target")


#weights and biases

weight = tf.Variable(tf.random_normal([num_hidden,num_classes]))
bias = tf.Variable(tf.random_normal([num_classes]))
cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
output, state = tf.nn.dynamic_rnn(cell,data, dtype=tf.float32)
output = tf.reshape(output, [num_classes,num_hidden])

with tf.name_scope('Softmax'):
     prediction = tf.nn.softmax(tf.matmul(output,weight) + bias)

# Add summary ops to collect data
w_h = tf.histogram_summary("weight", weight)
b_h = tf.histogram_summary("bias", bias)

with tf.name_scope('Cost'):
    cost = -tf.reduce_sum(target * tf.log(prediction))
    tf.scalar_summary("cost", cost)

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(.05).minimize(cost)

with tf.name_scope('accuracy'):
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float32'))
    tf.scalar_summary("accuracy", accuracy)


#init = tf.initialize_all_variables()
#merged_summary_op = tf.merge_all_summaries()

# Launch the graph
def Session():
    sess= tf.Session
    init = tf.initialize_all_variables()
    merged_summary_op = tf.merge_all_summaries()
    sess.run(init)
    writer = tf.train.SummaryWriter("/home/raisa/PycharmProjects/graphs", sess.graph_def)

    for i in range(epoch):
        epoch_loss = 0
        for j in range(1):
            inp, out = X, train_output
            inp = inp.reshape((1,10,1))
            j, c = sess.run([optimizer, cost], {data: inp, target: out})
        print "Epoch - ", str(i)
        epoch_loss += c
        print('Epoch', epoch ,'loss: {:3.1f}%'.format(epoch_loss*100))

        # Write logs for each iteration

    if i % 10 == 0:
        summary_str = sess.run(merged_summary_op, feed_dict={data: inp, target: out})
        writer.add_summary(summary_str, i * no_of_batches + batch_size)


Session()



