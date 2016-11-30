import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
import random

# generating signal
lorange = 1
hirange = 25
amplitude = np.random.uniform(-10, 10)
t = 784
random.seed()
tau = np.random.uniform(lorange, hirange)


def genrate_data(s):
    X = np.arange(100*(s-1),100*s)
    Y = amplitude * np.exp(-X / tau)
    return X, Y


# parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 100

# Network Parameters
n_input = 784  # data input (signal shape: 28*28)
n_classes = 2  # total classes (yes/no)
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
y = tf.placeholder(tf.float32, shape=(784,))  # signal
y_class = tf.placeholder(tf.float32, shape=(1,))  # labels
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


# creating wrappers
def conv2d(y, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    y = tf.nn.conv2d(y, W, strides=[1, strides, strides, 1], padding='SAME')
    y = tf.nn.bias_add(y, b)
    return tf.nn.relu(y)


def maxpool2d(y, k=2):
    # MaxPool2D
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Creating model

def conv_net(y, weights, biases, dropout):
    # Reshape input data
    y = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer 1
    conv1 = conv2d(y, weights['wc1'], biases['bc1'])
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
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(y, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y_class))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_class, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        X, Y = genrate_data(step)
        #X = np.reshape(X, (-1, 784))
        #Y = np.reshape(Y, (-1, 2))

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: X, y: Y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: X,
                                                              y: Y,
                                                              keep_prob: 1.})
            print "Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
        step += 1
    print "Finished training"

    # Calculate accuracy for 256 mnist test images
    # print "Testing Accuracy:", \
    # sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
    # y: mnist.test.labels[:256],
    # keep_prob: 1.})
