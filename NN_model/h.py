import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random


lorange = 1
hirange = 25
amplitude = np.random.uniform(-10, 10)
random.seed()
tau = np.random.uniform(lorange, hirange)
amplitude = 10
batchsize = 50
l = 10.0
step = 1


def genrate_data(s):
    X = np.arange(batchsize *(s-1),batchsize *s)
    Y = -amplitude * np.exp(-X / tau)
    return X, Y

y = tf.placeholder(tf.float32, shape=(784,))  # signal
label = tf.placeholder(tf.float32, shape=(1,))  # label
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

while step  < 10:
    X, Y = genrate_data(step)
    step += 1
    print "Finished training"
    print(X)
    print(Y)
    print (tau)
    plt.plot(X,Y)
plt.show()