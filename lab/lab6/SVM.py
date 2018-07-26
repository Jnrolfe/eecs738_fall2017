# coding: utf-8
"""
@filename: SVM.py
@author: James Rolfe
@date: 20170420
"""

import numpy as np


def weight_vector(x, y, lagrange_multiplier):
    """This function returns the primal weight vector.

    Args:
        x: a 2D array of floats
        y: a 1D array of -1's or +1's
        lagrange_multiplier: a 1D array of floats

    Returns:
        A numpy array of length equal to the number of columns in x.
    """
    w = [0] * x.shape[1] # initialize weights to 0
    
    # sum each aplhas * ys * column_vals 
    for j in range(len(w)):
        for i in range(x.shape[0]):
            w[j] += lagrange_multiplier[i] * y[i] * x[i][j]

    return np.array(w)


def support(x, y, w, b, tol=0.0015):
    """This function finds the indices for all of the support vectors.

    Args:
        x: a 2D array of floats
        y: a 1D array of -1's or +1's
        w: a 1D array of floats
        b: a float
        tol: a float representing how far away a support vector can be from the
             margin. This value should be between 0 and 1.

    Returns:
        A set of indices (int) that say which xs are support vectors.
    """
    support = set() # initialize set
    
    for i in range(x.shape[0]):
        # calculate the moment of the margin for each point
        moment = y[i] * (np.dot(w.T, x[i]) + b) - 1

        if((moment <= tol) and (moment >= (-1 * tol))):
            support.add(i)

    return support


def slack(x, y, w, b):
    """This function finds the indices for all of the slack vectors. Slack
    vectors are defined as misclassifed points.

    Args:
        x: a 2D array of floats
        y: a 1D array of -1's or +1's
        w: a 1D array of floats representing the primal weight vector.
        b: a float

    Returns:
        A set of indices (int) that say which xs are slack vectors.
    """
    w = np.array(w) # make w a numpy array to get access to dot product
    slack = set()

    for i in range(x.shape[0]):
        # calculate the moment of the margin for each point
        moment = y[i] * (np.dot(w.T, x[i]) + b) - 1
        
        # a moment with a negative value is misclassified
        if(moment < 0):
            slack.add(i)
    
    return slack


# hardcoded linearly inseparable data
inseparable = np.array([(2, 10, +1),
    (8, 2, -1),
    (5, -1, -1),
    (-5, 0, +1),
    (-5, 1, -1),
    (-5, 2, +1),
    (6, 3, +1),
    (7, 1, -1),
    (5, 2, -1)])

# hardcoded linearly separable data
separable = np.array([(-2, 2, +1),    
    (0, 8, +1),     
    (2, 1, +1),     
    (-2, -3, -1),   
    (0, -1, -1),    
    (2, -3, -1),    
    ])

# separate xs and ys out of the data
x1 = separable[:, 0:2]
y1 = separable[:, 2]
x2 = inseparable [:, 0:2]
y2 = inseparable [:, 2]

# hardcoded lagrange multipliers
lagrange_multiplier = np.zeros(len(x1))
lagrange_multiplier[4] = 0.34
lagrange_multiplier[0] = 0.12
lagrange_multiplier[2] = 0.22

# get the primal weight vector for the linearly separable data
w = weight_vector(x1, y1, lagrange_multiplier)
print('primal weight vector:')
print(w)

# hardcode intercept b
b = -0.2

# get the supports
s = support(x1, y1, w, b)
# print the supports
print('\nsupport vectors:')
print(s)

# hardcode new weights and intercept
w2 = [-.25, .25]
b2 = -.25
# get the slacks
s1 = slack(x2, y2, w2, b2)
# print slacks
print('\nslack vectors:')
print(s1)
