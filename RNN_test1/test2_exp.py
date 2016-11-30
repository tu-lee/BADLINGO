import numpy as np
import random
import math
from mlxtend.preprocessing import one_hot
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

batch_size = 1
no_of_batches = 1
epoch = 10

#creating random tau values for the signal
lorange= 1
resolution= 500
hirange= 1000
amplitude= np.random.uniform(-10,10)
#t= 0:100
t= np.linspace(0,100, num =101)
random.seed()

tau= np.array([int(math.ceil(np.random.uniform(lorange,hirange)))])
X= amplitude * np.exp(-t / tau)

#print X

#For every training input sequence,
# we generate an equivalent one hot encoded output representation.
train_input = X
train_output= np.array(one_hot([int(math.ceil(tau/resolution))]))
#print len(train_output)
#print len(train_input)


#splitting train and test data
#so far we have 1000 unique sequences
#NUM_EXAMPLES = 80
#test_input = train_input[NUM_EXAMPLES:]
#test_output = train_output[NUM_EXAMPLES:]  # everything beyond 10,000

#train_input = train_input[:NUM_EXAMPLES]
#train_output = train_output[:NUM_EXAMPLES]  # till 10,000

test_input = train_output
test_output = train_output
#building the model
#Placeholders will hold a fixed place for the inputs and labels
#Placeholders will be supplied with data later
#The dimension for data is 3 dimension in tensorflow so (batch_size, sequence length, input dimension)


train_input = tf.convert_to_tensor(train_input)
#train_input = tf.get_variable("train_input",shape = 101)
train_input = tf.reshape(train_input,[1,101,1])

#print ses.run(train_input)

train_output = tf.convert_to_tensor(train_output)
train_output = tf.reshape(train_output, [1,2])
ses = tf.Session()
print ses.run(train_output)

#print train_input1
data = tf.placeholder(tf.float32, shape= [batch_size,len(t),1])


train_output = tf.convert_to_tensor(train_output)
target = tf.placeholder(tf.float32, shape = train_output.get_shape())

print "input shape",train_input.get_shape()
print "output shape",train_output.get_shape()
print data


#creating the RNN cell
#For each LSTM cell that we initialise,
# we need to supply a value for the hidden dimension, or
# the number of units in the LSTM cell

num_hidden = 24
cell = rnn_cell.LSTMCell(num_hidden)
val, state = rnn.dynamic_rnn(cell, data, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
print tf.shape(val)
#last = tf.gather(val, int(val.get_shape()[0]) - 1)



#weights and biases

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))


#training
prediction = tf.nn.softmax(tf.matmul(val,weight) + bias)
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

for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = X[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    print "Epoch - ",str(i)
incorrect = sess.run(error,{data: test_input, target: test_output})
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()









