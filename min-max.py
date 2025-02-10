import numpy as np
import torch
import torch.nn as nn

# Create a vector 
v = np.array([1, 40, 2, -3])

# find and report the maximum and minimum values
minval = np.min(v)
maxval = np.max(v)

print('Min, Max: %g, %g' %(minval, maxval))

# now for the armin/max
minidx = np.argmin(v)
maxidx = np.argmax(v)

print('Min, max indices: %g, %g' %(minidx, maxidx)), print(' ')

# confirm

print(f'Min val is {v[minidx]}, max val is {v[maxidx]}')


# various minima in this matrix:

M = np.array([ [0,1,10],
               [20,8,5]
            ])

minidx1 = np.argmin(M)    # minimum from ENTIRE matrix
minidx2 =  np.argmin(M, axis=0)  # minimum of each column (aross rows)
minidx3 = np.argmin(M, axis=1)  # minimum of each row (across columns)

# print them out
print(M), print(' ') # reminder
print(minidx1)
print(minidx2)
print(minidx3)
