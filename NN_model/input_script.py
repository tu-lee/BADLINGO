import numpy as np
import tensorflow as tf
import random
from random import shuffle
import math

lorange= 1
resolution= 5
hirange= 10
amplitude= np.random.uniform(-10,10)
t= 100
random.seed()
tau=np.random.uniform(lorange,hirange)
x= amplitude * np.exp(-t / tau)


# classier(tau, resolution):
#str= '{0:02b}'.format(int(math.ceil(tau/resolution)))
#print str
#return str
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




print tf.shape(train_input)
#number of classes 2
#that is 1 and 0
#so target vector [1,0]

#n_labels= 2
#class_num= np.array([1,0])

#label_onehot = np.equal.outer(class_num, np.arange(n_labels)).astype(np.float)
#label_onehot= np.eye(n_labels)[class_num]

#ohm = np.zeros((class_num.shape[0], n_labels))
#empty one-hot matrix
#ohm[np.arange(class_num.shape[0]), class_num] = 1
#set target idx to 1




