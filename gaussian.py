'''
this module contains the gaussian function
author : Yu-Cheng Chung
email  : ycchung@ntnu.edu.tw
date   : 2023 13 Sep

dependencies:
    numpy
    matplotlib
'''
import numpy as np
import matplotlib.pyplot as plt
import transform

def gaussian(x:int|float|np.ndarray, mu:int|float = 0 , sigma:int|float = 1 )->int|float|np.ndarray:
    '''
    Gaussian function
    Args:
        x: the input of the gaussian function, can be a number or a np.ndarray
        mu: the mean, default is 0
        sigma: the standard deviation, default is 1
    Returns:
        the value of the gaussian function if x is a number
        the value of the gaussian function for each element in x if x is a np.ndarray
    '''
    return 1/(sigma * (2 * np.pi)**0.5) * np.exp(-0.5 * ((x - mu)/sigma)**2)


if __name__ == '__main__':
    #test the gaussian function
    vector=gaussian(np.arange(16),mu=8,sigma=2)
    t_vector=transform.normalize_prob_distribution(vector)
    plt.bar(range(16),t_vector)
    plt.plot(np.arange(16),vector)
