'''
@file: Question5-lab1.py
@author: James Rolfe
@updated: 20170130
'''

import random
import numpy as np
import matplotlib.pyplot as plt

# constant declaration
LOW_BOUND = 0
HIGH_BOUND = 1
# number of samples taken
NUM_OF_SAMPLES = [25, 10, 50, 100, 500, 1000]
# number of data points per sample
NUM_OF_DATA = [25, 10, 50, 100, 500, 1000]

# scale for exponential
EXP_SCALE = 1

np.random.seed(124)  # set the seed

#use the following function to test CLT.
#The following code generates samples using 'uniform',
#or samples from a uniform distribution.
'''
@args: a: int, b: int, num_of_data: int, num_of_samples: int
@post: prints to stdout and displays graph using matplotlib.pyplot
@returns: null
'''
def clt_uniform(a, b, num_of_data, num_of_samples):
    means = []

    #Here is a for loop to generate samples with low=a, high=b based on the given
    #numberOfSample and numberOfData, e.g. sample = numpy.random.uniform(...)
    for i in range(num_of_samples):
        sample = np.random.uniform(a, b, num_of_data)
        sample_mean = np.mean(sample)
        
        # append mean of sample
        means.append(sample_mean)

    # mean of samples' means is stored in the following line
    ans = np.mean(means)

    print('population mean for ' + str(num_of_data) + ' data points and ' 
            + str(num_of_samples) + ' samples is: ' + str(ans))
    plt.hist(means)
    plt.title('Hist given '+ str(num_of_data)+ ' data points and ' 
        + str(num_of_samples) + ' samples (UNIFORM)')
    plt.show()

'''
@args: a: int, num_of_data: int, num_of_samples: int
@post: prints to stdout and displays graph using matplotlib.pyplot
@returns: null
'''
def clt_exponential(a, num_of_data, num_of_samples):
    means = []

    for i in range(num_of_samples):
        sample = np.random.exponential(a, num_of_data)
        sample_mean = np.mean(sample)
        
        # append mean of sample
        means.append(sample_mean)

    # mean of samples' means is stored in the following line
    ans = np.mean(means)

    print('population mean for ' + str(num_of_data) + ' data points and ' 
            + str(num_of_samples) + ' samples is: ' + str(ans))
    plt.hist(means)
    plt.title('Hist given '+ str(num_of_data)+ ' data points and ' 
        + str(num_of_samples) + ' samples (EXPONENTIAL)')
    plt.show()

print('========== UNIFORM RESULTS ==========')

for x in NUM_OF_DATA:
    clt_uniform(LOW_BOUND, HIGH_BOUND, x, NUM_OF_SAMPLES[0])

for x in NUM_OF_SAMPLES:
    clt_uniform(LOW_BOUND, HIGH_BOUND, NUM_OF_DATA[0], x)

print('=== Test how many data points required for accurate'
       + ' mean estimate ===')
for i in range(4):
    clt_uniform(LOW_BOUND, HIGH_BOUND, i+1, 1000)

print('\n========== EXPONENTIAL RESULTS ==========')

for x in NUM_OF_DATA:
    clt_exponential(EXP_SCALE, x, NUM_OF_SAMPLES[1])

for x in NUM_OF_SAMPLES:
    clt_exponential(EXP_SCALE, NUM_OF_DATA[1], x)
