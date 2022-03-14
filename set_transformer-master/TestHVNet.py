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

# CMD code: bash run_test_HVNet.sh 7 model_M3_HVNet_5.pth test_data_M3_0.mat cuda
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

    def my_loss(output, pred):      # [bs, 100, 1]
        mask = output != 0
        output = output[mask]       # [bs*100]
        pred = pred[mask]           # [bs*100]
        loss = torch.mean(abs(output - pred)/output)

        print("output.shape", output.shape)
        print("pred.shape", pred.shape)
        print("loss", loss)

        return loss

    ## loading process for scio
    # solutionset = torch.from_numpy(data.get('Data')).float()
    # hv = torch.from_numpy(data.get('HVval')).float()
    # hv = torch.reshape(hv, (hv.shape[0],1,1))

    ## loading process for h5py
    solutionset = torch.from_numpy(np.transpose(data.get('Data'))).float()  # [dataset_num, data_num, M]
    hvc = torch.from_numpy(np.transpose(data.get('HVCval'))).float()  # [dataset_num, data_num]
    hvc = torch.reshape(hvc, (hvc.shape[0], hvc.shape[1], 1))  # [dataset_num, data_num, dim_output]
    print(f"solution set shape: {solutionset.shape}")
    print(f"hvc shape: {hvc.shape}")

    size = solutionset.shape[0]
    batch_size = 100

    dim_input = solutionset.shape[2]
    num_outputs = 1
    dim_output = 1

    model = DeepSet(device, dim_input, num_outputs, dim_output)     # HVNet
    model.load_state_dict(torch.load(os.path.join(path_dir, 'model', model_file)))
    model = model.to(device)
    print(f"Load model {model_file} done!")

    result = []
    dataloader = DataLoader(TensorDataset(solutionset, hvc), batch_size=batch_size, num_workers=1)
    loss_fn = nn.MSELoss()

    num_batches = len(dataloader)
    model.eval()
    test_loss = []
    start_time = time.time()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            input, output = X.to(device), y.to(device)  # [bs, 100, 3], [bs, 100, 1]

            # obtain pred_whole
            pred_whole = model.forward_allow_nan(input)  # [bs, 1, 1]
            # print('pred_whole:', pred_whole)
            # for each point in input (if not nan), obtain pred_without that is the predicted HV value without this point.
            # for 1 -> 100, set each column in the axis=1 in input to [nan, nan, nan]
            # and calculate pred_without [bs, 1, 1],
            # obtain pred [bs, 1, 1] as pred_whole - pred_without, for a batch
            pred = []
            for point_idx in range(input.shape[1]):
                # if these column points in this batch are all nan, just set pred_hvc as zeros [bs, 1, 1]
                # not a problem since there might be fully 100 points in 1 set.
                input_without = input.clone()
                input_without[:, point_idx, :] = float('nan')
                pred_without = model.forward_allow_nan(input_without)  # [bs, 1, 1]
                # print('pred_without:', pred_without)
                pred.append(pred_whole - pred_without)        # [bs, 1, 1]
            # cat all pred with axis=1 as the pred
            pred = torch.cat(pred, dim=1)        # [bs, 100, 1]
            # calculate loss on pred and output
            loss = my_loss(output, pred)
            test_loss.append(loss.item())
            #pred = model(X)
            #test_loss += my_loss(y, pred)
            result.append(pred.cpu().detach().numpy())       # num_batches * [bs, 100, 1]
    test_loss = np.mean(test_loss)
    end_time = time.time()
    #correct /= size
    #print(f"Avg loss: {test_loss:>8f} \n")
    print(f'Avg Loss and Time Used: {[float(test_loss), end_time-start_time]}')
    scio.savemat(os.path.join(path_dir, 'results', save_file),
                 {'result': np.concatenate(result, axis=0), 'loss': test_loss, 'time': end_time-start_time})
