'''
@filename: NaiveBayes.py
@author: james rolfe
@updated: 20170216
'''
from __future__ import division
from sklearn import datasets
import pandas as pd
import numpy as np
import math
import operator

# GLOBAL VARIABLE DECLARATION
# change these variables to change the outcome of the script!
TRAIN_CSV = "NaiveBayesTrain.csv"
TEST_CSV = "NaiveBayesTest.csv"

# Load the training and test dataset.
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

# print the training and test data
print("training data")
print(train)
print("\ntest data")
print(test)

# save the ground truth of the test data
groundtruth = test['target']


'''
This function returns the probability of value x using a gaussian distribution
@params: x:float, mean:float, std:float
@post: none
@returns: float less than 1
'''
def normal_dist_func(x, mean, std):
    c = 1/(math.sqrt((2*math.pi))*std)
    e = -((x-mean)**2)/(2*(std)**2)
    return c*math.exp(e)


'''
This function takes a testset of data in a data frame and a training data set as
df_temp in a data frame then calculates the conditional probability of each
data in the testset given the gaussian distribution of each column in the
training data
@params: testset:pandas.DataFrame, df_temp:pandas.DataFrame
@post: none
@returns: prob:array
'''
def calc_cnd_prob(testset, df_temp):
    # initialize array to save each row's probability, initializing to one is ok
    # since anything multiplied by 1 retained
    prob = [] 
    for i in testset.index:
        prob.append(1)

    # iterate through each column in the training data finding the mean and std
    # for each
    for col in df_temp:
        train_col_mean = np.mean(df_temp[col])
        train_col_std = np.std(df_temp[col])
        test_col = testset[col]
        
        # iterate over each column in the test data calculating the conditional
        # probability and multiplying the corresponding saved probability in the
        # prob array
        for i in range(len(test_col)):
            prob[i] = prob[i] * normal_dist_func(test_col[i], train_col_mean,
                    train_col_std)
    return prob 

prob_df = pd.DataFrame() # initialize the probability data frame
test = test.drop('target', axis = 1)

# bayes probability of the given class
# FIXME: Im not sure what this is used for since it is just a constant that is
# multiplying all the probabilities, this doesn't acutal change any predictions
probTarget = 1/len(train['target'].unique())
print("\nProbability target: " + "{0:.4f}".format(probTarget))

# For each label in the training dataset, we compute the probability of the 
# test instances.
for label in train['target'].unique():
    df_temp = train[train['target']==label]
    df_temp = df_temp.drop('target', axis = 1)
    testset = test.copy(deep=True)
    
    # get the probability matrix for each class definition and load the prob
    # data frame
    prob_df[label] = [(probTarget) * x for x in calc_cnd_prob(testset, df_temp)]

print('\nprob_df')
print(prob_df)

# define a list that stores each test data row's class prediction given the max
# of each row in the prob data frame. the index of the prediction array
# corresponds to the row it predicts for.
prediction = []
for i in prob_df.index:
    prediction.append(prob_df.loc[i,:].idxmax())
print('\nprediction list')
print(prediction)


'''
This function takes a prediction array full of class values and a ground truth
data frame full of class values and returns a confusion matrix data frame.
@params: prediction:array, ground_truth:pandas.DataFrame
@post: prints the conf matrix to stdout
@returns: confMatrix:pandas.DataFrame
'''
def confusionMatrix(prediction, ground_truth):
    # get the column names for the conf matrix
    names = list(train['target'].unique())
    names.reverse() # make the list so the conf matrix comes out as requested
    
    # initialize the data that will make up the conf matrix data frame
    conf_data = [[0 for i in range(len(names))] for j in range(len(names))]
    confMatrix = pd.DataFrame(conf_data, index=names, columns=names)

    # names the columns and indices of the conf matrix
    confMatrix.index.names = ['actual']
    confMatrix.columns.names = ['predicted']
    
    # load the data
    for i in range(len(prediction)):
        confMatrix.loc[ground_truth[i], prediction[i]] += 1

    print('\nconfusion matrix')
    print(confMatrix)

    return confMatrix


'''
This function calculates the trace of a pandas data frame. The dataframe is
assumed to be square.
@params: df:pandas.DataFrame
@post: none
@returns: float
'''
def df_trace(df):
    # sum the left-to-right diagnal of the data frame 
    return sum(df.loc[i,i] for i in range(len(df)))

conf = confusionMatrix(prediction, groundtruth)

# calculate the accuracy as the number of true positives plus true negatives
# over the total classifications. multiplied by 100 for percentage
accuracy = (float(df_trace(conf))/conf.sum().sum())*100
print('\naccuracy = ' + str(accuracy) + '%')
