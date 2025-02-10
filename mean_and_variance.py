import numpy as np

# create a list of numbers to compute the mean and variance of

x = [1,2,4,6,5,4,0]
n = len(x)

# compute the mean

mean1 = np.mean(x)
mean2 = np.sum(x) / n

# print them
print(mean1)
print(mean2)


# Variance

var1 = np.var(x)
var2 = (1/(n-1)) * np.sum( (x-mean1)**2)
var3 = np.var(x, ddof=1) # unbised


print(var1)
print(var2)
print(var3)

