import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random



#hyperparameters
Batch_size= 100
max_iteration= 50
learning_rate=1000
num_filt= 10



#training data
lorange= 1
hirange= 15
amplitude= 10
t= 10
random.seed()
tau=np.random.uniform(lorange,hirange)

def generate_data(randomsignal):
    X= np.arange(t)
    Y= amplitude*np.exp(-X/tauA)
    return X, Y


#tensors for input data

X= tf.placeholder(tf.float32, shape= [None, 10])
Y= tf.placeholder(tf.float32, shape= [None])
Y_class= tf.argmax(Y, dimension=1)

