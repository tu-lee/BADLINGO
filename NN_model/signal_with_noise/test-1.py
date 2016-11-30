import numpy as np
import matplotlib.pyplot as plt
import random

#Free induction decay is expressed with the function f(t)= ae-(iwt)-kt
#We will plot the real part of this function
#Real part is Re(f(t))= a(e-(kt)cos(wt))
#Rate constant k=1/tau
#angular frequency w
#pre exponential constant a

fig= plt.figure()

k = 0.05 #k=1/tau, bigger the tau , slower decay
w = 2
a = 5
time = 100
t= np.arange(0,time,1)
FID = a * (np.exp(-k*t))*(np.cos(w*t))
print FID

#adding white noise
mean = 0
std = 1
num_samples = 100
samples = .5 * np.random.normal(mean, std, size=num_samples)

signal= FID + samples

ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

ax1.plot(t, FID)
ax2.plot(t, samples)
ax3.plot(signal)
plt.show()


