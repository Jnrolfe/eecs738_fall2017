'''
@file: Question2-lab1.py
@author: James Rolfe
@updated: 20170129
'''

import random 

N = [1000, 10, 50, 100, 500, 1000]
genders = ('B', 'G')

random.seed(124) # set seed

for NUM_OF_FAMILIES in N:
    # init and populate sample space
    families = []
    for i in range(NUM_OF_FAMILIES):
        family = []
        family.append(random.choice(genders))
        family.append(random.choice(genders))
        families.append(family)

    # count the families with at least one son: oneson, 2 sons: twosons,... using 
    # the sample space created before
    one_son_cnt = 0  # at least one son
    first_son_cnt = 0  # first child is son
    two_sons_cnt = 0  # both children are boys
    two_daughters_cnt = 0  # both children are girls

    for family in families:
        if family == ['G', 'G']:
            two_daughters_cnt += 1
        if family == ['B', 'B']:
            two_sons_cnt += 1
            one_son_cnt += 1
            first_son_cnt += 1
        if family == ['B', 'G']:
            first_son_cnt += 1
            one_son_cnt += 1
        if family == ['G', 'B']:
            one_son_cnt += 1

    ''' # sanity check
    print("families with two daughters: " + str(two_daughters_cnt))
    print("families with two sons: " + str(two_sons_cnt))
    print("families with a son first: " + str(first_son_cnt))
    print("families with at least one son: " + str(one_son_cnt))
    '''

    two_sons_prob = "{0:.3f}".format(float(two_sons_cnt)/NUM_OF_FAMILIES)
    one_son_two_son_prob = "{0:.3f}".format(float(two_sons_cnt)/one_son_cnt)
    first_son_two_son_prob = "{0:.3f}".format(float(two_sons_cnt)/first_son_cnt)

    first_daughter_cnt = NUM_OF_FAMILIES - first_son_cnt
    temp_sum_1 = float(two_daughters_cnt) + two_sons_cnt
    temp_sum_2 = float(first_daughter_cnt) + first_son_cnt
    one_known_two_same_prob = "{0:.3f}".format(temp_sum_1/temp_sum_2)

    s_0 = ('===== NUMBER OF FAMILIES: ' + str(NUM_OF_FAMILIES) + ' =====\n')
    s_1 = ('probability of having two sons is: ' + two_sons_prob + '\n')
    s_2 = ('families that have at least one son, the probability of having two ' 
           + 'sons is: ' + one_son_two_son_prob + '\n')
    s_3 = ('families whose first child is a boy, the probability of having two '
           + 'sons is: ' + first_son_two_son_prob + '\n')
    s_4 = ('one child\'s gender is known, the probability of having two children '
           + 'of the same sex is: ' + one_known_two_same_prob + '\n')

    print(s_0 + s_1 + s_2 + s_3 + s_4)
