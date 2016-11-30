import numpy as np
from random import shuffle
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn


train_input = ['{0:020b}'.format(i) for i in range(2 ** 20)]
shuffle(train_input)
train_input = [map(int, i) for i in train_input]
ti = []
for i in train_input:
    temp_list = []
    for j in i:
        temp_list.append([j])
    ti.append(np.array(temp_list))
train_input = ti



#For every training input sequence,
# we generate an equivalent one hot encoded output representation.

train_output = []

for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count += 1
    temp_list = ([0] * 21)
    temp_list[count] = 1
    train_output.append(temp_list)

#splitting train and test data
#so far we have 2^20 (1,048,576) unique examples

NUM_EXAMPLES = 10000
test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:]  # everything beyond 10,000

train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES]  # till 10,000


#building the model
#Placeholders will hold a fixed place for the inputs and labels
#Placeholders will be supplied with data later
#The dimension for data is 3 dimension in tensorflow so (batch_size, sequence length, input dimension)

data = tf.placeholder(tf.float32, [None, 20,1])
target = tf.placeholder(tf.float32, [None, 21])

#creating the RNN cell
#For each LSTM cell that we initialise,
# we need to supply a value for the hidden dimension, or
# the number of units in the LSTM cell

num_hidden = 24
cell = rnn_cell.LSTMCell(num_hidden)
val, state = rnn.dynamic_rnn(cell, data, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

#weights and biases

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))


#training
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
cross_entropy = -tf.reduce_sum(target * tf.log(prediction))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)


#Calculating the error
mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

#Executing the model
init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)


#Giving data
batch_size = 1000
no_of_batches = int(len(train_input)/batch_size)
epoch = 5000
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    print "Epoch - ",str(i)
incorrect = sess.run(error,{data: test_input, target: test_output})
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()

