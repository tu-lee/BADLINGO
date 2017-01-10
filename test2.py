import tensorflow as tf
import numpy as np
import random
import math
from mlxtend.preprocessing import one_hot
from tensorflow.python.ops import rnn, rnn_cell


"""
Hyperparameters
"""
hm_epochs = 10
n_classes = 2
batch_size = 100
chunk_size = 100
n_chunks = 100
rnn_size = 128


"""
Parameters for generating input multiple exponential signals for input
"""

lorange= 1
resolution= 500
hirange= 1000
amplitude= np.random.uniform(-10,10)
t=100
no_t = 10000
no_tau = 10000

"""Input signals"""
for i in range(no_t):

    random.seed()
    train_input= np.random.uniform(lorange,hirange)
    X= amplitude * np.exp(-t / train_input)
    print(X)


"""Input labels"""
for l in range(no_tau):

    random.seed()
    tau = np.random.uniform(lorange, hirange)
    train_output = np.array(one_hot([int(math.ceil(tau / resolution))]))
    #print(train_output)


""" Input placeholders for signal and label"""
x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

"""Define RNN function"""
def recurrent_neural_network(x):

    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(0, n_chunks, x)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output

"""Training the network"""
def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < no_t:
                start = i
                end = i + batch_size
                batch_x = np.array(X[start:end])
                batch_y = np.array(train_output[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: x_train.reshape((-1, n_chunks, chunk_size)), y: y_train}))


train_neural_network(x)
