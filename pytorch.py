import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # The list of numbers
    z = [1, 2, 3]

    # compute the softmax result

    num = np.exp(z)
    den = np.sum(np.exp(z))
    sigma = num / den

    print(sigma) 
    print(np.sum(sigma))

    # repeat with some random integers

    z2 = np.random.randint(-5, high=15, size=25)

    num2 = np.exp(z2)
    den2 = np.sum(np.exp(z2))
    sigma2 = num2 / den2

    plt.plot(z2, sigma2, 'ko')
    plt.xlabel('original number (z2)')
    plt.ylabel('Softmaxified $\sigma$')

    plt.title('$\sum\sigma2$ = %g' %np.sum(sigma2))
    plt.show()


    ## Using pytorch

    softfun = nn.Softmax(dim=0)

    sigmaT = softfun(torch.Tensor(z2))

    print(sigmaT)





