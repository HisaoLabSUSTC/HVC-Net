import os
import sys

import numpy as np
import scipy.io as scio
import h5py
import torch
from models import *
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time

# CMD code: bash run_test.sh 7 model_whole_1.pth test_data_random_2.mat cuda
if __name__ == "__main__":
    path_dir = '/liaoweiduo/HVC-Net-datasets'  # for ubuntu server
    # path_dir = '//10.20.2.245/datasets/HVC-Net-datasets'  # for windows

    if len(sys.argv) is not 4:
        raise Exception('please specify model, test_data and device.')

    if sys.argv[3] == 'cpu':
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # model_file = 'model_50_50_100_1.pth'
    # model_file = 'model_tri_invertri_1.pth'
    # model_file = 'model_whole_1.pth'
    model_file = sys.argv[1]

    # test_file = 'test_data_tri_2.mat'
    # test_file = 'test_data_invtri_2.mat'
    # test_file = 'test_data_random_2.mat'
    test_file = sys.argv[2]

    save_file = f'result_{model_file[:-4]}_{test_file[:-4]}_{device}.mat'

    # Test
    path = os.path.join(path_dir, 'data', test_file)
    print(f'Test on: {test_file}')
    # data = scio.loadmat(path)
    data = h5py.File(path)

    def my_loss(output, pred):      # [bs, 100, 1]  contain nan
        mask = ~torch.isnan(output)
        output = output[mask]       # [bs*100]
        pred = pred[mask]           # [bs*100]
        loss = torch.mean(abs(output - pred)/output)
        return loss

    def CIR(output, pred, count=0, criteria='min'):      # [bs, 100, 1]  contain nan
        for x, y in zip(output, pred):  # [100, 1]
            if criteria == 'min':
                count = count + int(torch.argmin(x[~torch.isnan(x)]) == torch.argmin(y[~torch.isnan(y)]))
            elif criteria == 'max':
                count = count + int(torch.argmax(x[~torch.isnan(x)]) == torch.argmax(y[~torch.isnan(y)]))
            else:
                raise Exception('un-implemented CIR criteria. ')

        return count

    ## loading process for scio
    # solutionset = torch.from_numpy(data.get('Data')).float()
    # hv = torch.from_numpy(data.get('HVval')).float()
    # hv = torch.reshape(hv, (hv.shape[0],1,1))

    ## loading process for h5py
    solutionset = torch.from_numpy(np.transpose(data.get('Data'))).float()  # [dataset_num, data_num, M]
    hvc = torch.from_numpy(np.transpose(data.get('HVCval'))).float()  # [dataset_num, data_num]
    hvc = torch.reshape(hvc, (hvc.shape[0], hvc.shape[1], 1))  # [dataset_num, data_num, dim_output]

    size = solutionset.shape[0]
    batch_size = 100

    dim_input = solutionset.shape[2]
    num_outputs = 1
    dim_output = 1

    model = DeepSetHVC(device, dim_input, num_outputs, dim_output)
    model.load_state_dict(torch.load(os.path.join(path_dir, 'model', model_file)))
    model = model.to(device)
    print(f"Load model {model_file} done!")

    result = []
    dataloader = DataLoader(TensorDataset(solutionset, hvc), batch_size=batch_size, num_workers=1)
    loss_fn = nn.MSELoss()

    num_batches = len(dataloader)
    model.eval()
    test_loss = []
    CIR_min_count = 0
    CIR_max_count = 0
    start_time = time.time()

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # loss = []
            # for batch_idx in range(batch_size):
            #     input, output = X[batch_idx:batch_idx + 1], y[batch_idx:batch_idx + 1]  # [1, 100, 3] [1, 100, 1]
            #     mask = ~torch.isnan(input[0, :, 0])  # [100]
            #     input = input[:, mask == True]  # [1, 30, 3]
            #     output = output[:, mask == True]  # [1, 30, 1]
            #     pred = model(input)  # [1, 30, 1]
            #     loss.append(my_loss(output, pred))
            # loss = torch.mean(torch.stack(loss))
            # 每个set包含的valid point数不同，求mean之后再求mean，和直接所有的求mean是会有不同的。

            pred = model.forward_allow_nan(X)       # [bs, 100, 1]  contain nan
            loss = my_loss(y, pred)

            test_loss.append(loss.item())
            result.append(pred.cpu().detach().numpy())       # num_batches * [bs, 100, 1]

            # CIR_min
            CIR_min_count = CIR(y, pred, CIR_min_count, criteria='min')

            # CIR_max
            CIR_max_count = CIR(y, pred, CIR_max_count, criteria='max')

            # # each dataset
            # result_batch = []
            # for i in range(len(X)):
            #
            #     pop = X[i]        # [N, M]
            #     mask = ~torch.isnan(pop[:, 0])  # [100]
            #     pop = pop[mask == True, :]  # [30, 3]
            #
            #     if pop.shape[0] == 1:       # only contain 1 point.
            #         continue
            #
            #     pred = model(pop.unsqueeze(0))
            #     loss = my_loss(y[i][mask == True].unsqueeze(0), pred)
            #
            #     test_loss.append(loss.item())
            #     result_batch.append(y[i].cpu().detach().numpy())        # each pred[i] has different shapes.
            #
            #     CIR_min_count = CIR(y[i:i + 1], pred, CIR_min_count, criteria='min')
            #     CIR_max_count = CIR(y[i:i + 1], pred, CIR_max_count, criteria='max')
            #
            # result.append(np.stack(result_batch))

    test_loss = np.mean(test_loss)
    CIR_min = CIR_min_count / (int(size/batch_size) * batch_size)
    CIR_max = CIR_max_count / (int(size/batch_size) * batch_size)
    end_time = time.time()
    #correct /= size
    #print(f"Avg loss: {test_loss:>8f} \n")
    print(f'[Avg Loss, CIR min, CIR max, Time Used]: {[float(test_loss), CIR_min, CIR_max, end_time-start_time]}')
    scio.savemat(os.path.join(path_dir, 'results', save_file),
                 {'result': np.concatenate(result, axis=0),
                  'loss': test_loss, 'CIR_min': CIR_min, 'CIR_max': CIR_max, 'time': end_time-start_time})
