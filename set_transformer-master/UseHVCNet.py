"""
Toy example for illustrating how to use HVC-Net
"""

import numpy as np
from models import *

# Parameter settings
device = "cpu"
dim_input = 5 # Number of objectives
num_outputs = 1
dim_output = 1
num_block  = 10 # Number of K in HVC-Net

# Initialize HVC-Net and load the trained model
model = DeepSetHVC(device, dim_input, num_outputs, dim_output, num_blocks=num_block)
model.load_state_dict(torch.load('model_5_M5_10.pth', map_location=torch.device('cpu')))
print("Load Done!")

# Prepare a non-dominated solution set
set = np.array([(0.1,0.8,0.3,0.7,0.5),
                (0.8,0.4,0.2,0.2,0.5),
                (0.5,0.2,0.7,0.5,0.1)])

# Transform the solution set to tensor format
solutionset = torch.from_numpy(set).float()
solutionset = torch.reshape(solutionset,(1,solutionset.shape[0],solutionset.shape[1]))

# Use HVC-Net to handle the solution set and output the HVC approximations
print(model.forward(solutionset))
