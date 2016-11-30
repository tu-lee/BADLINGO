import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
import random
import math

lorange= 1
resolution= 5
hirange= 10
amplitude= np.random.uniform(-10,10)
t= 784
random.seed()
tau=np.random.uniform(lorange,hirange)
x1= amplitude * np.exp(-t / tau)
X= x1

#x2= np.vectorize(x1)
#X= np.arange(x1)




def classifier(str):
    str = '{0:02b}'.format(int(math.ceil(tau/resolution)))
    return str


print tau, classifier(str)


#parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 10
dropout= 0.25


#tf Graph input
x= tf.placeholder(tf.float32, shape= [None, 784]) #signal
y= tf.placeholder(tf.float32, shape= [None, 2]) #labels
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


#creating wrappers
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


#Creating model

def conv_net(x, weights, biases, dropout):
    # Reshape input data
    x = tf.reshape(x, shape=[-1, 28, 28, 1])


    # Convolution Layer 1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer 2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, 2]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([2]))
}



# Construct model
pred = conv_net(x, weights, biases, keep_prob)



# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)



# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



# Initializing the variables
init = tf.initialize_all_variables()



# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = str(batch_size)
        #X= np.reshape(X,(-1,784))
        #str = np.reshape(classifier, (-1, 2))


        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y,
                                                              keep_prob: 1.})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
        step += 1
    print "Finished training"

    # Calculate accuracy for 256 mnist test images
    #print "Testing Accuracy:", \
        #sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      #y: mnist.test.labels[:256],
                                      #keep_prob: 1.})