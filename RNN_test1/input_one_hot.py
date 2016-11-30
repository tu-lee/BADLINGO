import numpy as np
import random
import math
from mlxtend.preprocessing import one_hot
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn



#creating random tau values for the signal
lorange= 1
resolution= 500
hirange= 1000
amplitude= np.random.uniform(-10,10)
t= np.linspace(0,100, num =101)
print t
random.seed()

train_input= np.array([int(math.ceil(np.random.uniform(lorange,hirange)))])
X= amplitude * np.exp(-t / train_input)

#For every training input sequence,
# we generate an equivalent one hot encoded output representation.
train_output= np.array(one_hot([int(math.ceil(train_input/resolution))]))
print train_output
print train_input
print tf.shape(train_input)
print tf.shape(train_output)
