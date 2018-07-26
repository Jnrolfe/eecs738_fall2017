import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm, uniform

np.random.seed(124)  # set the seed

# Generate a sample of 100000 data points from 1 to 10 with uniform 
# distribution in the following line: 
uniform_sample = np.random.uniform(0, 10, 100000)

# Plot the distribution (Hint: using pandas pd)
s = pd.Series(uniform_sample, index=range(100000))
df = pd.DataFrame(s)
df[0].plot(kind='hist')
plt.title('100000 uniform samples between [0, 10]')
plt.xlabel('value')
plt.ylabel('frequency')
plt.show()

# Cutoff the 25% quantile (Hintt: using from scipy.stats import uniform)
# get the first quartile interval of a uniform distribution
quantile_25 = uniform.interval(0.25, loc=0, scale=10)
df[0].plot(kind='hist', xlim=quantile_25)
plt.title('25% quantile of uniform distribution between [0, 10]')
plt.xlabel('value')
plt.ylabel('frequency')
plt.show()

# Calculate the pdf and cdf with N(0,1) at value 0 (Hint: from scipy.stats
# import norm)
print('pdf at value 0 of N(0,1): ' + str(norm(0, 1).pdf(0)))
print('cdf at value 0 of N(0,1): ' + str(norm(0, 1).cdf(0)))

#explain the following 5 lines of code (What are they doing?)
mydata = np.random.randn(1000)
kde = gaussian_kde( mydata )
y = np.linspace(min(mydata), max(mydata), num=100)
plt.plot( y, kde(y) ,label="estimation",color="red")
plt.title("non-parametric pdf")
plt.show()

