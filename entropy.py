import numpy as np 
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    # Entropia 
    x = [.25, .75] # lista de probabilidades (x1, x2, x3, ... xn)

    H = 0
    for p in x:
        H -= p * np.log(p)  # -= pois (p * np.log(p)) retorna um valor negativo

    print('Entropy: ' + str(H))

    ## cross entropy

    p = [1, 0]
    q = [0.25, 0.75]

    H = 0

    for i in range(len(p)):
        H -= p[i]*np.log(q[i])
    
    print ('Cross Entropy: ' + str(H))


import torch 
import torch.nn.functional as F 

# nota : os imputs tem que ser tensores

q_tensor = torch.Tensor(q)
p_tensor = torch.Tensor(p)

F.binary_cross_entropy(p_tensor, q_tensor)

