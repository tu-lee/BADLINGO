import numpy as np
import random
import math
from mlxtend.preprocessing import one_hot
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
import matplotlib.pyplot as plt


num_hidden = 12
batch_size = 1
no_of_batches = 1
epoch = 10
num_classes=10
lorange= 1
resolution= 500
hirange= 1000
amplitude= np.random.uniform(-10,10)
#t= np.linspace(0,10, num =10)
t=np.arange(10)
random.seed()

tau= np.random.uniform(lorange,hirange)
X= np.array([amplitude * np.exp(-t / tau)])#.reshape((1,10,1))
print 'X', X

#------------Sorting out the input
train_input = X
#train_input = tf.convert_to_tensor(train_input)
train_input = tf.reshape(train_input,[batch_size,len(t),1])
print 'ti', train_input


#------------sorting out the output
train_output= [int(math.ceil(tau/resolution))]
print 'tau', train_output
train_output= one_hot(train_output, num_labels=10)
print 'onehot', train_output

#test data, keep it same as train data for now
test_input = train_output
test_output = train_output

#tf graph input
data = tf.placeholder(tf.float32, shape= [batch_size,len(t),1], name= "Data")
target = tf.placeholder(tf.float32, shape = [batch_size, num_classes], name= "Target")


#creating the RNN cell
#For each LSTM cell that we initialise,
# we need to supply a value for the hidden dimension, or
# the number of units in the LSTM cell


#weights and biases
def recurrent_neural_network(data):

    weight = tf.Variable(tf.random_normal([num_hidden,num_classes]))
    bias = tf.Variable(tf.random_normal([num_classes]))
    cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
    output, state = tf.nn.dynamic_rnn(cell,data, dtype=tf.float32)
    output = tf.reshape(output, [num_classes,num_hidden])

    outputt = tf.nn.softmax(tf.matmul(output,weight) + bias)
    return outputt


def train_neural_network(data):
    prediction = recurrent_neural_network(data)
    cost = -tf.reduce_sum(target * tf.log(prediction))

    optimizer = tf.train.GradientDescentOptimizer(.05).minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float32'))

    sess = tf.InteractiveSession()
    init = tf.initialize_all_variables()
    sess.run(init)

    #epoch_loss =0
    for i in range(epoch):
        epoch_loss = 0
        for j in range(1):
            inp, out = X, train_output
            inp = inp.reshape((1,10,1))
            j, c = sess.run([optimizer, cost], {data: inp, target: out})
        print "Epoch - ", str(i)
        epoch_loss += c
        print('Epoch', epoch ,'loss: {:3.1f}%'.format(epoch_loss*100))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({data: inp, target: out}))

train_neural_network(data)



