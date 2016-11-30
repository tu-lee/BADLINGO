import numpy as np
import random
from mlxtend.preprocessing import one_hot
import matplotlib.pyplot as plt


num_classes=10


amplitude1= np.random.uniform(-10,10)
t= np.arange(5)
random.seed()
N = [1,3,5,7,9]
tau1= [np.random.random() for i in (N)]
X1= amplitude1 * np.exp(-t / tau1)
print 'X1', X1



amplitude2= np.random.uniform(-10,10)
random.seed()
M = [0,2,4,6,8]
tau2= [np.random.random() for i in (M)]
X2= amplitude2 * np.exp(-t / tau2)
print 'X2', X2



#------------sorting out the output
label1= one_hot(N, num_labels=10)
print 'signal1', label1

label2= one_hot(M, num_labels=10)
print 'signal2', label2


fig= plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.plot(t, X1)
ax2.plot(t, X2)
plt.show()

