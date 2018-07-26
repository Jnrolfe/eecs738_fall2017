'''
@file: Question4-lab1.py
@author: James Rolfe
@updated: 20170130
'''

import numpy as np
# import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

np.random.seed(124) # set seed

# np.random.normal(<mean>, <std_deviation>, <num_samples>)
sample_1 = np.random.normal(10, 4, 1)
sample_1000 = np.random.normal(10, 4, 1000)

sample_1000_mean = np.mean(sample_1000)
sample_1000_variance = np.var(sample_1000)

s = ('The mean of 1000 samples from a normal distribution with mean 10 and '
    + 'standard deviation of 4 is: ' + str(sample_1000_mean) + '\nit\'s '
    + 'variance is: ' + str(sample_1000_variance))
print(s)
