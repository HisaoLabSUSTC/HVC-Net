"""
Calculate HVC by Monte Carlo method
"""

import numpy as np
import scipy.io as scio
import math
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

def MC_HVC(data_set, exclusive_index, num_sample, reference_point, is_maximum):
    """
    Calculate the HVC by Monte Carlo method
        :param data_set: A single data set, in [point_index, point_dimension] format
        :param exclusive_index: The exclusive data point, which is the index of the point that want to calcualte the HVC
        :param num_sample: The number of sample in Monte Carlo
        :param reference_point: The reference point
        :param is_maximum: Whether is maximum problem
        :return: The HVC of exclusive_index
    """
    (num_data, dimension) = np.shape(data_set)
    u = np.zeros(dimension)
    if is_maximum is True:
        dominance_matrix = data_set >= data_set[exclusive_index]
    else:
        dominance_matrix = data_set <= data_set[exclusive_index]
    exclusive_point = data_set[exclusive_index]
    for i in range(dimension):
        temp = (dominance_matrix[:, i] == 0) & (np.sum(dominance_matrix[:, [j for j in range(i)] + [j for j in range(i + 1, dimension)]], axis=1) == dimension - 1)
        candidates = data_set[temp==1, :]
        if len(candidates) == 0:
            u[i] = reference_point[i]
        else:
            if is_maximum is True:
                u[i] = max(candidates[:, i])
            else:
                u[i] = min(candidates[:, i])
    points = np.random.rand(num_sample, dimension)
    if is_maximum is True:
        points = points * abs(exclusive_point - u) + u
    else:
        points = points * abs(exclusive_point - u) + exclusive_point
    data_set = np.delete(data_set, exclusive_index, axis=0)
    miss = 0
    for i in range(num_sample):
        if is_maximum is True:
            dominance_check = (data_set >= points[i, :])
        else:
            dominance_check = (data_set <= points[i, :])
        if data_set.shape[0] == 0:      # origin dataset only 1 point
            dominance_check = np.array([[False for _ in range(dimension)]])
        if max(np.sum(dominance_check, axis=1)) == dimension:
            miss+=1
    return ((num_sample - miss) / num_sample) * np.prod(abs(exclusive_point - u))

def MC2(data_set, reference_point, k, num_sample):
    """
    Calculate HVC for all points in a data set
        :param data_set:
        :param reference_point:
        :param k:
        :param num_sample:
    """
    # Only work when M > 2
    (N, dimension) = np.shape(data_set)
    F_min = np.min(data_set, axis=0)
    S = np.random.rand(num_sample, dimension) * np.matlib.repmat(reference_point-F_min, num_sample, 1) + np.matlib.repmat(F_min, num_sample, 1)
    PdS = np.zeros((N, num_sample), dtype=int)
    dS = np.zeros(num_sample, dtype=int)
    for i in range(N):
        x = np.sum(np.matlib.repmat(data_set[i], num_sample, 1) - S <= 0, axis=1) == dimension
        PdS[i][x == 1] = 1
        dS[x == 1] = dS[x == 1] + 1
    alpha = np.zeros(N)
    alpha[0] = np.prod(k/N)
    F = np.zeros(N)
    for i in range(N):
        a = PdS[i]
        b = dS[a] - 1
        b[b < 0] = 0
        F[i] = sum(alpha[b])
    F = F * np.prod(reference_point - F_min)/num_sample
    return F


if __name__ == "__main__":
    path = 'data-HVC.mat'
    data = scio.loadmat(path)
    solutionset = data.get('Data')
    hvc = data.get('HVCval')

    index = 1
    set = solutionset[index]

    mask = ~np.isnan(set)
    set = set[mask[:, 0] == True]

    solutionindex = 10
    approx = MC_HVC(set, solutionindex, 10000, np.ones([1,3]), False)
    print(approx)

    print(hvc[index][solutionindex])

