# Neural Network Hessian Eigenvalue Spectrum Density 
## IE510 Applied Non-Linear Programming | Spring 2022 | UIUC

Implementation of hessian eigendensity estimation for neural networks and 
generative adversarial networks. 

## Contributors
1. Abhinav Garg (garg19@illinois.edu)
2. Parth Gera (gera2@illinois.edu)

## Install dependencies
Here is an example of create environ **from scratch** with `anaconda`

```sh
# create conda env
conda create --name eigendens python=3.9
conda activate eigendens
# install dependencies
pip install -r requirements.txt
```

## GAN Training
This repository uses the ResNet32 GAN architecture 
proposed by Sun et al. in their paper "Towards a 
Better Global Loss Landscape of GANs" on CIFAR-10
dataset 
([repo](https://github.com/AilsaF/RS-GAN))

To train the model, specify the arguments in `rsgan.py` 
and run `python rsgan.py`

The training progress can be monitored from the directory
created by the script with name corresponding to the specified
arguments.

## Hessian Eigendensity Estimation
To estimate the hessian eigendensity spectrum use `hessian_spectrum.py`.
The script uses the trained GAN network to compute eigenvalues and 
eigenvectors of the hessian at different epochs.

Specify the following arguments in the script and 
run `python hessian_spectrum.py`.
```
num_eigenthings   # int, number of top eigenvalues/eigenvectors to compute
model             # str, gen or dis model for the file saved
use_gpu           # bool, True or False 
mode              # str, 'power_iter' or 'lanczos' to specify the method to estimate hessian
norm              # bool, True or False for the file saved
epochs            # list, list of epochs to run the method for
save_path         # str, directory of the trained models 
```

The script saves a pickle file in the `results` directory. 
Use `analysis.ipynb` to visualize the spectrum for further analysis.