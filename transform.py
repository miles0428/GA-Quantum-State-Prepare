'''
this module is used to transform the state vector and probability distribution
author : Yu-Cheng Chung
email  : ycchung@ntnu.edu.tw
date   : 2023 12 Sep

dependencies:
    numpy

'''
import numpy as np

def normalize_prob_distribution(distribution : np.ndarray[float|int])->np.ndarray[int|float]:
    '''
    Calculate the probability distribution of the given distribution
    make sure the sum of the distribution is 1
    Args:
        distribution: the distribution
    Returns:
        the probability distribution
    '''
    #check if distribution have minus value or have complex value
    if np.any(distribution < 0):
        raise ValueError('distribution should not have minus value')
    elif np.any(np.iscomplex(distribution)):
        raise ValueError('distribution should not have complex value')
    return (distribution/np.sum(distribution)).astype(float)

def normalize_state_vector(vector : np.ndarray[float|int|complex])->np.ndarray[int|float|complex]:
    '''
    Calculate the state vector of the given state vector
    make sure the inner product of the state vector and its conjugate is 1
    Args:
        state_vector: the state vector
    Returns:
        the state vector
    '''
    product = np.conjugate(vector) * vector
    return vector/abs(np.sum(product))

def prob2statevector(prob_distribution : np.ndarray[int|float])->np.ndarray[int|float|complex]:
    '''
    Calculate the state vector of the given distribution
    Args:
        prob_distribution: the probability distribution
    Returns:
        the state vector
    '''
    #check if prob_distribution have minus value or have sum not equal to 1 or have complex value
    if np.any(prob_distribution < 0):
        raise ValueError('prob_distribution should not have minus value')
    elif np.sum(prob_distribution) != 1:
        raise ValueError('prob_distribution should sum to 1')
    elif np.any(np.iscomplex(prob_distribution)):
        raise ValueError('prob_distribution should not have complex value')
    return np.sqrt(prob_distribution)

def statevector2prob(state_vector : np.ndarray[float|int|complex])->np.ndarray[float|int]:
    '''
    Calculate the probability distribution of the given state vector
    Args:
        state_vector: the state vector
    Returns:
        the probability distribution
    '''
    #check if state_vector has been normalized
    state_vector = np.array(state_vector)
    if abs(np.sum(np.conjugate(np.array(state_vector)) * np.array(state_vector)) - 1) > 1e-5:
        raise ValueError('state_vector should be normalized')
    degger = np.conjugate(state_vector)
    return abs(degger * state_vector).astype(float)

