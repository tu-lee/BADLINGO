import numpy as np
import matplotlib.pyplot as plt
import math


a = 10
t = np.arange(2048)
nsets = 10

values1 = np.random.uniform(1,5)
values2 = np.random.uniform(6,10)
values3 = np.random.uniform(11,15)
values4 = np.random.uniform(16,20)
values5 = np.random.uniform(21,25)
values6 = np.random.uniform(26,30)
values7 = np.random.uniform(31,35)
values8 = np.random.uniform(36,40)
values9 = np.random.uniform(41,45)
values10 = np.random.uniform(46,50)

candidates = [values1, values2, values3, values4, values5, values6, values7, values8, values9, values10]


def signal():
    tau1 = np.random.choice(candidates)
    tau2 = np.random.choice(candidates)
    tau3 = np.random.choice(candidates)
    tau4 = np.random.choice(candidates)
    tau5 = np.random.choice(candidates)
    y = a * np.exp(-t /tau1) + a * np.exp(-t /tau2) + a * np.exp(-t /tau3) + a * np.exp(-t /tau4) + a * np.exp(-t /tau5)
    return y

for i in range(nsets):
    signalset = []
    signalset.append(signal())
    print(signal())

for i in range (nsets):

    plt.plot(t, signal())
    plt.figure(i)
    plt.show()

