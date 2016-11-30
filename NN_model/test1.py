import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random


# %% Let's create some toy data
n_observation=100
lorange= 1
hirange= 10
amplitude= np.random.uniform(-10,10 )
t= 10
random.seed()
tau=np.random.uniform(lorange,hirange)
x_= np.arange(t, n_observation)
y_= amplitude*np.exp(-x_/tau)

plt.plot(x_,y_)

# %% tf.placeholders for the input and output of the network. Placeholders are
# variables which we need to fill in when we are ready to compute the graph.
x_input = tf.placeholder(tf.float32, shape=(10,))
y_input = tf.placeholder(tf.float32, shape=(10,))

# %% We will try to optimize min_(W,b) ||(X*w + b) - y||^2
# The `Variable()` constructor requires an initial value for the variable,
# which can be a `Tensor` of any type and shape. The initial value defines the
# type and shape of the variable. After construction, the type and shape of
# the variable are fixed. The value can be changed using one of the assign
# methods.
weight_1 = tf.Variable(tf.truncated_normal([10,10], stddev= .1))
bias_1 = tf.Variable(.1)
Y_pred = tf.add(tf.mul(x_input, weight_1), bias_1)

# %% Loss function will measure the distance between our observations
# and predictions and average over them.
cost = tf.reduce_sum(tf.pow(Y_pred - y_input, 2)) / (n_observation - 1)

# %% if we wanted to add regularization, we could add other terms to the cost,
# e.g. ridge regression has a parameter controlling the amount of shrinkage
# over the norm of activations. the larger the shrinkage, the more robust
# to collinearity.
# cost = tf.add(cost, tf.mul(1e-6, tf.global_norm([W])))

# %% Use gradient descent to optimize W,b
# Performs a single step in the negative gradient
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# %% We create a session to use the graph
n_epochs = 1000
with tf.Session() as sess:
    # Here we tell tensorflow that we want to initialize all
    # the variables in the graph so we can use them
    sess.run(tf.initialize_all_variables())

    # Fit all training data
    prev_training_cost = 0.0
    for epoch_i in range(n_epochs):
        for (x, y) in zip(x_, y_):
            sess.run(optimizer, feed_dict={x_input: x_, y_input: y_})

        training_cost = sess.run(
            cost, feed_dict={x_input: x_, y_input: y_})
        print(training_cost)

        if epoch_i % 20 == 0:
            plt.plot(x_, Y_pred.eval(
                feed_dict={X: xs}, session=sess),
                    'k', alpha=epoch_i / n_epochs)

            plt.plot()

        # Allow the training to quit if we've reached a minimum
        if np.abs(prev_training_cost - training_cost) < 0.000001:
            break
        prev_training_cost = training_cost


#fig.show()
#plt.waitforbuttonpress()