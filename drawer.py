'''
This file is used to draw the results from the GA

dependencies:
    gene.py
    GA.py
    numpy
    matplotlib

results structure:
    result = [[fidelity, depth, theta], [fidelity, depth, theta], ...]

genes structure:
    genes = [[gene], [gene], ...]

'''
import numpy as np
import matplotlib.pyplot as plt
from gene import Gene_Circuit
from GA import get_prob_distribution
import os

def draw_prob_distribution(gene : list, 
                           theta : list|np.ndarray, 
                           num_qubit : int, 
                           target_distribution : list|np.ndarray, 
                           filename:str) -> None:
    '''
    Draw the probability distribution of a circuit and the target distribution
    Args:
        gene: a list of 0-10
        theta: a list of theta
        num_qubit: number of qubits
        target_distribution: the target distribution
        filename: the filename of the picture
    Returns:
        None
    '''
    circuit = Gene_Circuit(gene, num_qubit).circuit
    prob_distribution = get_prob_distribution(circuit, theta)
    plt.clf()
    plt.bar(range(2**num_qubit), prob_distribution)
    plt.plot(np.arange(2**num_qubit), target_distribution, label='target_distribution')
    plt.ylim(0,0.4)    
    plt.savefig(filename)

def load_results_from_file(filename:str)->np.ndarray:
    '''
    Load the results from a file
    Args:
        filename: the filename of the file
    Returns:
        result: the result
    '''
    result = np.load(filename, allow_pickle=True)
    return result

def load_genes_from_file(filename:str)->np.ndarray:
    '''
    Load the genes from a file
    Args:
        filename: the filename of the file
    Returns:
        gene: the gene
    '''
    gene = np.load(filename)
    return gene

#{generation_number: [gene, gene, gene, ...]}
def draw_gene_circuit_from_result(path : str , 
                                  expriement : str, 
                                  num_qubit : int, 
                                  generation_number : int)->None:
    '''
    Draw the circuit of the smallest depth gene for each generation
    Args:
        path: the path of the file
        expriement: the name of the expriement
        num_qubit: number of qubits
        generation_number: number of generation
    Returns:
        None
    '''
    genes = {}
    for i in range(generation_number) :
        filename = f'{path}/{expriement}/{i}st_generation/10_smallest_depth_gene.npy'
        gene = load_genes_from_file(filename)
        genes[i] = gene
    os.makedirs(f'{path}/{expriement}/smallest_circuit',exist_ok=True)
    for i in range(generation_number):
        Gene_Circuit(genes[i][0], num_qubit).draw(output='mpl', filename=f'{path}/{expriement}/smallest_circuit/generation_{i}_smallest_depth_gene.png')

def draw_prob_distribution_from_result(path : str, 
                                       expriement : str, 
                                       target_distribution : list|np.ndarray, 
                                       num_qubit :int, 
                                       generation_number : int)->None:
    '''
    Draw the probability distribution of the smallest depth gene for each generation
    Args:
        path: the path of the file
        expriement: the name of the expriement
        target_distribution: the target distribution
        num_qubit: number of qubits
        generation_number: number of generation
    Returns:
        None
    '''
    genes = {}
    for i in range(generation_number) :
        filename = f'{path}/{expriement}/{i}st_generation/10_smallest_depth_gene.npy'
        gene = load_genes_from_file(filename)
        genes[i] = gene
    os.makedirs(f'{path}/{expriement}/smallest_distribution',exist_ok=True)
    results = {}
    for i in range(generation_number) :
        filename = f'{path}/{expriement}/{i}st_generation/10_smallest_depth_result.npy'
        result = load_results_from_file(filename)
        results[i] = result
    # print(results)
    for i in range(generation_number):
        if i == 1:
            continue
        theta = results[i][0,2]
        gene = genes[i][0]

        draw_prob_distribution(gene, theta, num_qubit, target_distribution, 
                               f'{path}/{expriement}/smallest_distribution/generation_{i}_smallest_depth_gene_prob_distribution.png')

def draw_fidelity_change_from_result(path : str, 
                                     expriement : str, 
                                     generation_number : int)->None:
    '''
    Draw the fidelity change with generation
    Args:
        path: the path of the file
        expriement: the name of the expriement
        generation_number: number of generation
    Returns:
        None
    '''
    results = {}
    for i in range(generation_number) :
        filename = f'{path}/{expriement}/{i}st_generation/10_smallest_depth_result.npy'
        result = load_results_from_file(filename)
        results[i] = result
    fidelity_change = []
    for i in range(generation_number):
        fidelity_change.append(sum(results[i][:,0])/len(results[i][:,0]))
    plt.clf()
    plt.plot(range(generation_number), fidelity_change)
    plt.savefig(f'{path}/{expriement}/fidelity_change.png')

def draw_depth_change_from_result(path : str, 
                                  expriement : str, 
                                  generation_number : int)->None:
    '''
    Draw the depth change with generation
    Args:
        path: the path of the file
        expriement: the name of the expriement
        generation_number: number of generation
    Returns:
        None
    '''
    results = {}
    for i in range(generation_number) :
        filename = f'{path}/{expriement}/{i}st_generation/10_smallest_depth_result.npy'
        result = load_results_from_file(filename)
        results[i] = result
    depth_change = []
    for i in range(generation_number):
        depth_change.append(sum(results[i][:,1])/len(results[i][:,1]))
    plt.clf()
    plt.plot(range(generation_number), depth_change)
    plt.savefig(f'{path}/{expriement}/depth_change.png')


if __name__ == '__main__':
    path = 'data'
    expriement = 'tset'
    num_qubit = 3
    generation_number = 30
    target_distribution = np.array([0.7, 0.7,0,0,0,0,0,0])
    draw_depth_change_from_result(path, expriement, generation_number)
    draw_fidelity_change_from_result(path, expriement, generation_number)
    draw_prob_distribution_from_result(path, expriement, target_distribution, num_qubit, generation_number)
    draw_gene_circuit_from_result(path, expriement, num_qubit, generation_number)
