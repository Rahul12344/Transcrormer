from sklearn.decomposition import PCA
import numpy as np
import torch

def covariance_matrix(X_train):
    dset = torch.stack(X_train).cpu().detach().numpy()
    nsamples, nx, ny = dset.shape
    d2_train_dataset = dset.reshape((nsamples,nx*ny))
    return np.cov(d2_train_dataset)

def covariance_matrix_estimate(X_train):
    return torch.mm(X_train.t(), X_train) / (X_train.shape[0] - 1)

