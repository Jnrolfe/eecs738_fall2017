
# coding: utf-8

"""
@filename: EM.py
@author: James Rolfe
@date: 20170414
"""

import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from scipy.stats import norm
from sys import maxint

from scipy.stats import multivariate_normal as multi_norm

#Please complete the following Expectation Maximization code, implementing a batch EM is sufficient for this lab

#set a random seed, remember to set the correct seed if you want to use another command for seeding
np.random.seed(124)

# we have *two* clusters. Note that the covariance matrices are diagonal
mu = [0, 6]
sig = [ [3, 0], [0, 4] ]

muPrime = [6, 0]
sigPrime = [ [5, 0], [0, 2] ]

#Generate samples of type MVN and size 100 using mu/sigma and muPrime/sigmaPrime. 
samp1 = np.random.multivariate_normal(mu, sig, 100)
x1, y1 = samp1[:,0], samp1[:,1]
samp2 = np.random.multivariate_normal(muPrime, sigPrime, 100)
x2, y2 = samp2[:,0], samp2[:,1]

x = np.concatenate((x1, x2))
y = np.concatenate((y1, y2))
# assign half the data to label 1 and the other half to label 2
# this is the correct labeling
labels = ([1] * 100) + ([2] * 100)

#Convert the data which now includes 'x' and 'y' to a dataframe. 
data = {'x': x, 'y': y, 'label': labels}
df = pd.DataFrame(data=data)


# Load the data and inspect it
df.head()


# Initial guess for mu, sigma, and alpha which are initially bad, α = probability of class asignment. ∑α=1 for k=1 to K.
initialGuess = { 'mu': [2,2], 'sig': [ [1, 0], [0, 1] ], 'muPrime': [5,5], 'sigPrime': [ [1, 0], [0, 1] ], 'alpha': [0.4, 0.6]}


# Compute the posterior with the help of computing the pdf, e.g. using norm.pdf
def posterior(val, mu, sig, alpha):
    '''posteriors'''
    prob = alpha * multi_norm.pdf(val, mean=mu, cov=sig)
    return list(prob)


# The E-step, estimate w, this w is the "soft guess" step for the class labels. You have to use the already defined posteriors in this step.
def expectation(dataFrame, parameters):
    '''This function uses the posteriors to estimate w.'''
    # get the probability that each point is in cluster 1
    w1 = posterior(dataFrame[['x','y']].values, parameters['mu'].values,
          list(parameters['sig'].values), parameters['alpha'].values[0])
    # get the probability that each point is in cluster 2
    w2 = posterior(dataFrame[['x','y']].values, parameters['muPrime'].values,
          list(parameters['sigPrime'].values), parameters['alpha'].values[1])
    
    # assign each point to a cluster dependent upon its prob
    for i in range(dataFrame.shape[0]):
        if w1[i] > w2[i]:
            dataFrame.loc[i, 'label'] = 1
        else:
            dataFrame.loc[i, 'label'] = 2

    return dataFrame

# The M - step: update estimates of alpha, mu, and sigma
def maximization(dataFrame, parameters):
    # get the points that are marked as cluster 1
    c1_points = dataFrame[dataFrame['label'] == 1]
    # get the points that are marked as cluster 2
    c2_points = dataFrame[dataFrame['label'] == 2]
    # get the percentage of points that belong to cluster 1 (alpha[0])
    c1_perc = float(len(c1_points)) / dataFrame.shape[0]
    # get the percentage of points that belong to cluster 2 (alpha[1])
    c2_perc = float(len(c2_points)) / dataFrame.shape[0]

    # update each parameter given the new markings
    parameters['alpha'] = [c1_perc, c2_perc]
    parameters['mu'] = [c1_points['x'].mean(), c1_points['y'].mean()]
    parameters['muPrime'] = [c2_points['x'].mean(), c2_points['y'].mean()]
    parameters['sig'] = [[c1_points['x'].std(), 0], [0, c1_points['y'].std()]]
    parameters['sigPrime'] = [[c2_points['x'].std(), 0], [0, c2_points['y'].std()]]

    return parameters

# Check Convergence, define your convergence criterion. You can define a new function for this purpose or just check it in the loop. You will have to use this function at the end of each while/for loop's EM iteration to check whether we have reached "convergence" or not. So to test for convergence, we can calculate the log likelihood at the end of each EM step (e.g. model fit with these parameters) and then test whether it has changed “significantly” (defined by the user, e.g. it should be something similar to: if(loglik.diff < 1e-6) ) from the last EM step. If it has, then we repeat another step of EM. If not, then we consider that EM has converged and then these are our final parameters.

# Iterate until convergence: with E-step, M-step, checking our etimates of mu/checking whether we have reached convergece, and updating the parameters for the next iteration. This part of the code should print a figure for *each* iteration, the *final* parameters, and #iterations. The final outcome that you have to submit is your EM code and a .pdf report. Your report should have the plots for **each** iteration, your **explanation on the convergence criterion you used based on last paragraph's explanations**, your final parameters, and the general flow of your code.



# loop until the parameters converge

iters = 0
params = pd.DataFrame(initialGuess)
converged = False

while(not converged):
    iters += 1
    converged = True

    # save the old labels to check for convergence
    old_df = df.copy(True)

    # E-step
    est_df = expectation(df, params)

    # M-step
    new_params = maximization(est_df, params)

    # check for convergence if no points have changed labels
    old_labels = list(old_df['label'].values)
    new_labels = list(est_df['label'].values)

    for new_label, old_label in zip(new_labels, old_labels):
        if new_label != old_label:
            converged = False

    # print parameters for each iteration
    print(new_params)   

    # update labels and parameters for the next iteration
    params = new_params

    # plot the clusters
    fig = plt.figure()
    plt.scatter(est_df['x'], est_df['y'], 20, c=est_df['label'])
    plt.title('Cluster 1 (blue) vs Cluster 2 (red)')
    plt.show()

print('total iters: ' + str(iters))
print('final params: ')
print(params)
