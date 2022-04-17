# HVC-Net: Deep Learning based Hypervolume Contribution Approximation

### Folder explanations
- *Figures* This folder contains all the experimental results in the paper.  
- *generateData* This folder contains the codes for generating training and testing soluiton sets used in the paper.  
- *models* This folder contains the trained HVC-Net models, which can be used directly in the future.  
- *set_transformer-master* This folder contains the codes for HVC-Net, the point-based method, and the line-based method.   

### How to use HVC-Net
In folder *set_transformer-master*, we provide `UseHVCNet.py` as an example. To use HVC-Net, we first initialize `DeepSetHVC` model, and then load the trained model. After that, we can input any solution set to the model and get the approximated hypervolume contributions.
