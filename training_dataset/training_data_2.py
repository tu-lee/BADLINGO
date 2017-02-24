import numpy as np
import matplotlib.pyplot as plt
import math
import  random


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

T1 = np.random.choice(candidates)
T2 = np.random.choice(candidates)
T3 = np.random.choice(candidates)
T4 = np.random.choice(candidates)
T5 = np.random.choice(candidates)


def random_tau(length):
    dna = [values1, values2, values3, values4, values5, values6, values7, values8, values9, values10]
    sequence = ''
    for i in range(length):
        sequence += random.choice(dna)
    return sequence


def signal():
    y = a * np.exp(-t / T1) + a * np.exp(-t / T2) + a * np.exp(-t / T3) + a * np.exp(-t / T4) + a * np.exp(-t / T5)
    return y

for i in range(nsets):
    sequenceset = []
    for i in range(2048):
        length = 5
        sequenceset.append(random_tau(length))

    for sequence in sequenceset:
        print (sequence)



plt.plot(t, signal())
plt.show()
