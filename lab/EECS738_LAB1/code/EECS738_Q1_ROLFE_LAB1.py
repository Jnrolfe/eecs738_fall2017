'''
@file: Question1-lab1.py
@author: James Rolfe
@updated: 20170129
'''

import numpy as np

np.random.seed(124) # set seed

# define sample spaces
ages = ['A<30', '30<=A<=50', 'A>50']
parties = ['D','R']

# size of samples
N = [100, 1000, 10000]

'''
@args: tuple_list: array of tuples, samp_size: int
@post: print to stdout
@return: null
'''
def print_prob_table(tuple_list, samp_size):
    samp_size = float(samp_size)

    # init counts
    d_30_cnt = 0 # democrat and age less than 30
    d_30_50_cnt = 0 # democrat and age between 30 and 50 inclusive
    d_50_cnt = 0 # democrat and age greater than 50
    r_30_cnt = 0 # republican and age less than 30
    r_30_50_cnt = 0 # republican and age between 30 and 50 inclusive
    r_50_cnt = 0 # republican and age greater than 50

    # add counts
    for data in tuple_list:
        if data == ('A<30', 'D'):
            d_30_cnt += 1
        elif data == ('30<=A<=50', 'D'):
            d_30_50_cnt += 1
        elif data == ('A>50', 'D'):
            d_50_cnt += 1
        elif data == ('A<30', 'R'):
            r_30_cnt += 1
        elif data == ('30<=A<=50', 'R'):
            r_30_50_cnt += 1
        elif data == ('A>50', 'R'):
            r_50_cnt += 1

    # calculate probabilities and format as string of 2 percision
    d_30_prob = "{0:.2f}".format(d_30_cnt/samp_size)
    d_30_50_prob = "{0:.2f}".format(d_30_50_cnt/samp_size)
    d_50_prob = "{0:.2f}".format(d_50_cnt/samp_size)
    r_30_prob = "{0:.2f}".format(r_30_cnt/samp_size)
    r_30_50_prob = "{0:.2f}".format(r_30_50_cnt/samp_size)
    r_50_prob = "{0:.2f}".format(r_50_cnt/samp_size)

    table = ("          |    D   |   R    \n"
             "   A<30   |  "+d_30_prob+"  |  "+r_30_prob+"  \n"
             "30<=A<=50 |  "+d_30_50_prob+"  |  "+r_30_50_prob+"  \n"
             "   A>50   |  "+d_50_prob+"  |  "+r_50_prob+"  \n")
    print("joint probability table for sample size: "+str(int(samp_size)))
    print(table)

for i in range(len(N)):
    samp_size = N[i]
    
    age_sample = np.random.choice(ages, samp_size, p=[0.3, 0.3, 0.4])
    party_sample = np.random.choice(parties, samp_size, p=[0.45, 0.55])

    print_prob_table(zip(age_sample, party_sample), samp_size)


