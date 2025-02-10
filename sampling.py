import numpy as np
import matplotlib.pyplot as plt

# create a list of numbers to compute the mean and variance of

x = [1,2,4,6,5,4,0,-4,5,-2,6,10,-9,1,3,-6]
n = len(x)

# compute the population mean

popmean = np.mean(x)

# compute a sample mean

sample = np.random.choice(x, size=5, replace=True)
sampmean = np.mean(sample)

# print them
print(popmean)
print(sampmean)

# compute lots of samples means

# number of experiments to run

nExperiments = 10000

# run the experiment!
sampleMeans = np.zeros(nExperiments)
for i in range(nExperiments):

    # step 1: draw a sample
    sample = np.random.choice(x, size=5, replace=True)

    # step 2: compute its mean
    sampleMeans[i] = np.mean(sample)

# show the results as a histogram

plt.hist(sampleMeans, bins=40, density=True)
plt.plot([popmean, popmean], [0, .3], 'm--')
plt.ylabel('count')
plt.xlabel('Sample mean')
plt.show()