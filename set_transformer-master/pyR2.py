"""
A program to calcualte HVC by proposed R2 method
"""

import numpy as np
import math
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import pyMC
import time
import scipy.io as scio
from math import gamma

def read_data(name, variable_name):
    mat_file = sio.loadmat(name)
    data = mat_file[variable_name]
    return data


def R2HVC(data_set, weight_vector_grid, exclusive_index, reference_point, is_maximize):
    """
    Calculate HVC of a signle point by proposed R2 method
    Parameters
    ----------
    data_set: Single data set
    weight_vector_grid: The weight vector grid in R2 method
    exclusive_index: The index of point that want to calculate HVC by R2 method
    reference_point: The reference point of R2
    is_maximize: Whether is maximize problem
    Returns
    -------
    A single HVC value of exclusive_index point
    """
    (num_weight_vector, dimenstion) = np.shape(weight_vector_grid)
    exclusive_point = data_set[exclusive_index, :]
    data_set_exclusive = np.delete(data_set, exclusive_index, axis=0)
    temp1 = np.min(abs(exclusive_point - reference_point) /
                   weight_vector_grid, axis=1)
    y = 0
    for i in range(num_weight_vector):
        if is_maximize is True:
            temp = (exclusive_point - data_set_exclusive) / \
                weight_vector_grid[i, :]
        else:
            temp = (data_set_exclusive - exclusive_point) / \
                weight_vector_grid[i, :]
        x = np.min(np.max(temp, axis=1))
        y = y + math.pow(min(x, temp1[i]), dimenstion)

    score = y * np.pi ** (dimenstion / 2) / (dimenstion * num_weight_vector * 2 ** (dimenstion - 1) * gamma(dimenstion / 2))
    #return y/num_weight_vector
    return score


def R2_least_contributor(data_set, weight_vector_grid, reference_point, is_maximize):
    """
    The method to find out the least HVC by R2 method
    Parameters
    ----------
    data_set:
    weight_vector_grid:
    reference_point:
    is_maximize:
    Returns
    -------
    The index of least HVC in the data set
    """
    (number_points, dimension) = np.shape(data_set)
    HVC = []
    for i in range(number_points):
        HVC.append((i, R2HVC(data_set, weight_vector_grid,
                             i, reference_point, is_maximize)))
    HVC = sorted(HVC, key=lambda x: x[1])
    return HVC[0][0]


def Double_R2_least_contributor(data_set, weight_vector_grid_small, weight_vector_grid_large, reference_point, is_maximize):
    """
    Find out the least HVC by apply R2 method twice.
    The first calculation using a small weight vector grid to choose half of candidates
    The second calculation using a large weight vector gird to precisely calculate the HVC
    and find the least contributor.
    Parameters
    ----------
    data_set:
    weight_vector_grid_small:
    weight_vector_grid_large:
    reference_point:
    is_maximize:
    Returns
    -------
    The index of least HVC contributor by R2 method
    """
    (number_points, dimension) = np.shape(data_set)
    HVC = []
    for i in range(number_points):
        HVC.append((i, R2HVC(data_set, weight_vector_grid_small,
                             i, reference_point, is_maximize)))
    HVC = sorted(HVC, key=lambda x: x[1])

    HVC_ac = []
    for i in [HVC[j][0] for j in range(int(number_points * 0.5))]:
        HVC_ac.append(
            (i, R2HVC(data_set, weight_vector_grid_large, i, reference_point, is_maximize)))
    HVC_ac = sorted(HVC_ac, key=lambda x: x[1])

    return HVC_ac[0][0]


def generate_WV_grid(num_of_vectors, dimension):
    """
    A method to generate weight vector grid
    Parameters
    ----------
    num_of_vectors:
    dimension:
    Returns
    -------
    numpy matrix [vector_index, vector_dimension]
    """
    mu = np.zeros((dimension))
    sigma = np.eye(dimension)
    R = np.random.multivariate_normal(mu, sigma, num_of_vectors)
    V = abs(R/np.sqrt(np.sum(np.square(R), axis=1))[:, None])
    return V


def get_weighted_vectors(M, H):
    """
    A method to generate weight vector grid
    Parameters
    ----------
    M: The number of objects
    H: A parameter that influence the number of weight vector
    Returns
    -------
    numpy matrix, every row is a weight vector
    """
    comb = [i for i in range(1, M + H)]
    weight_matrix = []
    comb = list(itertools.combinations(comb, M - 1))
    for space in comb:
        weight = []
        last_s = 0
        for s in space:
            w = (((s - last_s) - 1) / H)
            last_s = s
            weight.append(w)
        weight.append(((M + H - last_s) - 1) / H)
        weight_matrix.append(weight)
    V = np.array(weight_matrix)
    V = abs(V/np.sqrt(np.sum(np.square(V), axis=1))[:, None])
    return V


def plot_3D(EP):
    # Print 3D graph of EP
    fig = plt.figure(figsize=(9, 9), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='z', labelsize=20)
    # plt.xlim(0, 1.2)
    # plt.ylim(0, 1.2)
    # ax.set_zlim(0, 1.2)
    plt.xlabel("Minimize $f_1(x)$", fontsize=25, labelpad=15)
    plt.ylabel("Minimize $f_2(x)$", fontsize=25, labelpad=15)
    ax.set_zlabel("Minimize $f_3(x)$", fontsize=25, labelpad=15)
    temp = np.array(EP)
    ax.scatter(temp[:, 0], temp[:, 1], temp[:, 2], c='b')
    ax.view_init(20, 45)
    plt.show()


if __name__ == "__main__":
    path = 'data-HVC.mat'
    data = scio.loadmat(path)
    solutionset = data.get('Data')
    hvc = data.get('HVCval')

    index = 1
    set = solutionset[index]

    mask = ~np.isnan(set)
    set = set[mask[:, 0] == True]

    solutionindex = 20

    vectors = generate_WV_grid(1000, 3)
    approx = R2HVC(set, vectors, solutionindex, np.ones([1,3]), False)
    print(approx)

    print(hvc[index][solutionindex])