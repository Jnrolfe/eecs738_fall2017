'''
@file: Question3-lab1.py
@author: James Rolfe
@updated: 20170130
'''

import numpy as np
# import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

np.random.seed(124) # set seed

NUM_TRIAL = 10 # number of trials
PROB = 0.5 # probability of success for each trial
NUM_TEST = 340 # number of tests in each experiment
N = 5 # number of experiments

def run_experiment(num_exp):
    sample = np.array([]) # array to hold 

    for i in range(num_exp):
        # get np.array of samples from binomial distribution
        sample = np.append(sample, np.random.binomial(NUM_TRIAL, PROB,
            NUM_TEST))

    zero_cnt = sum(sample == 0) # how many test scores of zero
    one_cnt = sum(sample == 1) # how many test scores of one
    two_cnt = sum(sample == 2) # how many test scores of two
    
    # make histogram
    counts, bins, patches = plt.hist(sample, bins=NUM_TRIAL+1, range=[0,
        NUM_TRIAL], align='mid')

    # correct x-axis so the values are in the middle of each bar
    xlocs = np.array(0.5 * np.diff(bins) + bins[:-1])
    xvals = range(NUM_TRIAL + 1)
    plt.xticks(xlocs, xvals)

    # histogram labels
    plt.xlabel('Questions Answered Correctly')
    plt.ylabel('Frequency of Score')
    plt.title('Histogram for ' + str(num_exp) + ' experiments')
    plt.grid(False)

    plt.show()

    s = ('Probability of scoring a zero in ' + str(num_exp) 
        + ' experiments: ' + str(float(zero_cnt)/(num_exp*NUM_TEST)))
    ss = ('Probability of scoring a two or less in ' + str(num_exp) 
        + ' experiments: ' + str(float(zero_cnt + one_cnt + two_cnt)/(num_exp*NUM_TEST)))
    print(s)
    print(ss)

run_experiment(1)
run_experiment(10)
