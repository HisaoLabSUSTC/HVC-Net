import os
import sys

import numpy as np
import scipy.io as scio
import h5py
import time

from pyMC import MC_HVC
from pyR2 import generate_WV_grid, R2HVC

# CMD code: python TestMCR2.py 3
if __name__ == "__main__":
    path_dir = '/liaoweiduo/HVC-Net-datasets'  # for ubuntu server
    # path_dir = '//10.20.2.245/datasets/HVC-Net-datasets'  # for windows

    if len(sys.argv) == 2:
        M = int(sys.argv[1])
        # 1-19 seeds
        test_files = [f'test_data_M{M}_{seed}.mat' for seed in range(0, 10)]
    if len(sys.argv) == 3:
        M, seed = int(sys.argv[1]), int(sys.argv[2])
        test_files = [f'test_data_M{M}_200_{seed}.mat' for seed in range(seed, seed+1)]
        # test on seed 1 to find proper range
    if len(sys.argv) == 4:
        M, seed0, seed1 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        test_files = [f'test_data_M{M}_{seed}.mat' for seed in range(seed0, seed1)]
        # test on seed [seed0, seed1) to find proper range
    else:
        M = 3
        test_files = [f'test_data_M{M}_{seed}.mat' for seed in range(0, 10)]


    for test_file in test_files:
        save_file = f'result_MC_R2_{test_file[:-4]}.mat'

        # Test
        path = os.path.join(path_dir, 'data', test_file)
        print(f'Test on: {test_file}')
        # data = scio.loadmat(path)
        data = h5py.File(path)

        def my_loss(output, pred):  # [bs, 100, 1]  contain nan
            mask = ~np.isnan(output)
            output = output[mask]  # [bs*100]
            if len(output) != len(pred):
                pred = pred[mask]  # [bs*100]
            loss = np.mean(np.abs(output - pred) / output)
            return loss

        def CIR(output, pred, count=0, criteria='min'):  # [bs, 100, 1]  contain nan
            for x, y in zip(output, pred):  # [100, 1]
                if criteria == 'min':
                    count = count + int(np.argmin(x[~np.isnan(x)]) == np.argmin(y[~np.isnan(y)]))
                elif criteria == 'max':
                    count = count + int(np.argmax(x[~np.isnan(x)]) == np.argmax(y[~np.isnan(y)]))
                else:
                    raise Exception('un-implemented CIR criteria. ')

            return count

        ## loading process for scio
        # solutionset = torch.from_numpy(data.get('Data')).float()
        # hv = torch.from_numpy(data.get('HVval')).float()
        # hv = torch.reshape(hv, (hv.shape[0],1,1))

        ## loading process for h5py
        solutionset = np.transpose(data.get('Data'))  # [dataset_num, data_num, M]
        hvc = np.transpose(data.get('HVCval'))  # [dataset_num, data_num]
        hvc = np.reshape(hvc, (hvc.shape[0], hvc.shape[1], 1))    # [dataset_num, data_num, 1]

        L, N, M = solutionset.shape
        # sampleNum = np.arange(100, 2001, 100)   # [20,]   [100, 200, ..., 2000]
        # lineNum = np.arange(10, 201, 10)        # [20,]   [10, 20, ..., 200]
        # sampleNum = np.arange(10, 201, 10)   # [20,]   [10, 20, ..., 200]
        # lineNum = np.arange(1, 21, 1)        # [20,]   [1, 2, ..., 20]
        # sampleNum = np.arange(10, 201, 10)   # [20,]   [10, 20, ..., 200]
        # lineNum = np.arange(10, 201, 10)   # [20,]   [10, 20, ..., 200]
        sampleNum = np.arange(5, 101, 5)   # [20,]   [5, 10, ..., 100]
        lineNum = np.arange(5, 101, 5)   # [20,]   [5, 10, ..., 100]

        Time_MC = np.zeros(len(sampleNum))
        Loss_MC = np.zeros(len(sampleNum))
        CIRmin_MC = np.zeros(len(sampleNum))
        CIRmax_MC = np.zeros(len(sampleNum))
        Time_R2 = np.zeros(len(lineNum))
        Loss_R2 = np.zeros(len(lineNum))
        CIRmin_R2 = np.zeros(len(lineNum))
        CIRmax_R2 = np.zeros(len(lineNum))

        reference_point = [1.0 for _ in range(M)]

        for k in range(len(sampleNum)):
            print(f'MC {k}')
            loss = []
            CIR_min_count = 0
            CIR_max_count = 0
            CIR_total = 0
            start_time = time.time()
            for i in range(L):
                pop = solutionset[i]        # [N, M]
                mask = ~np.isnan(pop[:, 0])  # [100]
                pop = pop[mask == True, :]  # [30, 3]

                if pop.shape[0] == 1:       # only contain 1 point.
                    continue

                hvcmc = np.array([MC_HVC(pop, index, sampleNum[k], reference_point, is_maximum=False)
                         for index in range(len(pop))])
                hvcmc = hvcmc.reshape(hvcmc.shape[0], 1)

                loss.append(my_loss(hvc[i], hvcmc))

                # CIR_min
                CIR_min_count = CIR(hvc[i][np.newaxis, :], hvcmc[np.newaxis, :], CIR_min_count, criteria='min')

                # CIR_max
                CIR_max_count = CIR(hvc[i][np.newaxis, :], hvcmc[np.newaxis, :], CIR_max_count, criteria='max')

                CIR_total = CIR_total + 1

            loss = np.mean(loss)
            CIR_min = CIR_min_count / CIR_total
            CIR_max = CIR_max_count / CIR_total
            end_time = time.time()

            Loss_MC[k] = loss
            CIRmin_MC[k] = CIR_min
            CIRmax_MC[k] = CIR_max
            Time_MC[k] = end_time - start_time
            print(f'loss {loss}, CIR min {CIR_min}, CIR max {CIR_max}, time {end_time - start_time}')

        for k in range(len(lineNum)):
            print(f'R2 {k}')
            loss = []
            CIR_min_count = 0
            CIR_max_count = 0
            CIR_total = 0
            start_time = time.time()
            for i in range(L):
                pop = solutionset[i]        # [N, M]
                mask = ~np.isnan(pop[:, 0])  # [100]
                pop = pop[mask == True, :]  # [30, 3]

                if pop.shape[0] == 1:       # only contain 1 point.
                    continue

                hvcr2 = np.array([R2HVC(pop, generate_WV_grid(lineNum[k], M), index, reference_point, is_maximize=False)
                         for index in range(len(pop))])
                hvcr2 = hvcr2.reshape(hvcr2.shape[0], 1)

                loss.append(my_loss(hvc[i], hvcr2))

                # CIR_min
                CIR_min_count = CIR(hvc[i][np.newaxis, :], hvcr2[np.newaxis, :], CIR_min_count, criteria='min')

                # CIR_max
                CIR_max_count = CIR(hvc[i][np.newaxis, :], hvcr2[np.newaxis, :], CIR_max_count, criteria='max')

                CIR_total = CIR_total + 1

            loss = np.mean(loss)
            CIR_min = CIR_min_count / CIR_total
            CIR_max = CIR_max_count / CIR_total
            end_time = time.time()

            Loss_R2[k] = loss
            CIRmin_R2[k] = CIR_min
            CIRmax_R2[k] = CIR_max
            Time_R2[k] = end_time - start_time
            print(f'loss {loss}, CIR min {CIR_min}, CIR max {CIR_max}, time {end_time - start_time}')

        # save
        scio.savemat(os.path.join(path_dir, 'results', save_file), {
            'Loss': Loss_MC,  'CIR_min': CIRmin_MC,
            'CIR_max': CIRmax_MC, 'Time': Time_MC,
            'Loss1': Loss_R2,  'CIR_min1': CIRmin_R2,
            'CIR_max1': CIRmax_R2,
            'Time1': Time_R2})
