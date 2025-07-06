import torch
import torch.nn as nn
import torch.nn.functional as funs
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

class D1WLSNetwork(nn.Module):
    """ Create D1WLS NUC Algorithm Implementation
        The following implementation utilized on the basis of the original matlab implementation
        https://github.com/LifangzhouSia/1D-Weighted-Least-Square-Destriping-for-Uncooled-Infrared-Images
        verifyPyTorchOutputs.m function is used to verify the PyTorch model using the outputs of the Matlab model
        Total absolute difference between two implementation is about 0.005, which is 6x10^{-8} per pixel on average
    """
    
    def __init__(self, lamda=40, titer=3):
        super(D1WLSNetwork, self).__init__()
        
        self.lamda = lamda
        self.titer = titer
        
    def forward(self, inImage):
        
        # implementation of d1_WLS_Destriping algorithm (inputs: CROP_SIZE x CROP_SIZE)
        images = inImage.to(torch.float64)
        HEIGHT = images.shape[0]
        WIDTH = images.shape[1]
        
        # initial image into[0, 1] range
        initialFactor = images.view(-1).max(dim=0).values
        L = images / initialFactor
        
        # create weights
        weights = torch.ones((1, WIDTH), device=L.device, dtype=L.dtype)
        
        # start iterations
        u = L.clone()
        for t in range(self.titer):
        
            Dire = self.edgeIndicator(u, 0.3)
            sigma_n = self.get_Sigma(u)
            
            # horizontal edge-preserving smoothing stage
            for k in range(HEIGHT):
                u[k:(k+1), :] = self.fGHS(u[k:(k+1), :], weights, self.lamda, 2 * sigma_n)
            
            # do weighted filtering
            u = self.weightedLocalLinearFilter(u, L, Dire, np.floor(HEIGHT / 6).astype(int), 0.01)

        # output FPN
        return ((L - u) * initialFactor).to(inImage.dtype)

    def fGHS(self, g, w, lambda_, sigma_n):
        """Gaussian Homotopy Smoothing"""
        hei, wid = g.shape
        g = g.flatten()
        w = w.flatten()

        dg = torch.diff(g)
        dg = torch.exp(-dg ** 2 / (sigma_n ** 2))
        
        # zero tensor
        ztensor = torch.tensor([0], device=dg.device, dtype=dg.dtype)
        
        # prepare diagonals for the tri-diagonal matrix
        dga = -lambda_ * torch.cat([ztensor, dg]).cpu().numpy()
        dgc = -lambda_ * torch.cat([dg, ztensor]).cpu().numpy()
        dg = (w + lambda_ * (torch.cat([ztensor, dg]) + torch.cat([dg, ztensor]))).cpu().numpy()
        f = (w * g).cpu().numpy()
        
        # solve the linear system A*u = b where A is a tri-diagonal matrix defined by dga, dg, dgc
        u = self.linearInverseOperator(dga, dg, dgc, f)

        return torch.tensor(u, dtype=g.dtype).reshape(hei, wid)

    def linearInverseOperator(self, a, b, c, f):
        """
        Solves a tridiagonal system Ax = f where the subdiagonal (a), diagonal (b), and
        superdiagonal (c) of the matrix A are given, along with the right-hand side vector f.
        """
        DL = len(a)
        u = np.zeros_like(f)
    
        # forward sweep
        c[0] = c[0] / b[0]
        f[0] = f[0] / b[0]
        for k in range(1, DL):
            c[k] = c[k] / (b[k] - c[k - 1] * a[k])
            f[k] = (f[k] - f[k - 1] * a[k]) / (b[k] - c[k - 1] * a[k])
    
        # backward substitution
        u[DL - 1] = f[DL - 1]
        for k in range(DL - 2, -1, -1):
            u[k] = f[k] - c[k] * u[k + 1]
    
        return u
    
    def weightedLocalLinearFilter(self, x, y, w, r, eps):
        """ Performs weighted local linear filtering on two input images x and y """
        # x, y, w of size HEIGHT x WIDTH
        ww = self.verticalBoxFilter(w, r)
        wx = self.verticalBoxFilter(w * x, r) / ww
        wy = self.verticalBoxFilter(w * y, r) / ww
        xwy = self.verticalBoxFilter(w * x * y, r) / ww
        wxx = self.verticalBoxFilter(w * x * x, r) / ww
        wyy = self.verticalBoxFilter(w * y * y, r) / ww
    
        a = (xwy - wx * wy + eps) / (wxx - wx * wx + eps)
        b = wy - wx * a
        
        # normalization factors
        mean_a = self.verticalBoxFilter(a, r)
        mean_b = self.verticalBoxFilter(b, r)
    
        return (y - mean_b) / mean_a
    
    def edgeIndicator(self, L, xi, r=5):
        # perform vertical box filtering on the transposed image
        Lt = torch.transpose(L, 0, 1)
        
        # get the transposed vertically filtered images
        m1 = self.verticalBoxFilter(Lt, r)
        m2 = self.verticalBoxFilter(Lt*Lt, r)
    
        # calculate variance in transposed dimensions
        Var = torch.transpose(m2 - torch.pow(m1, 2), 1, 0)
    
        # compute the mean of variance for each batch
        m = Var.mean()
    
        # calculate directional response
        return torch.exp(-Var / (xi * m)) + 1e-10
    
    def verticalBoxFilterUnnormalized(self, L, r=5):
        """ Applies a vertical box filter to the input tensor L: HEIGHT x WIDTH"""
        kernel_size = (2 * r + 1, 1)
        kernel = torch.ones((1, 1, *kernel_size), device=L.device, dtype=L.dtype)
        
        # add batch and channel dimensions for conv2d
        input = L.unsqueeze(0).unsqueeze(0)
        filtered = funs.conv2d(input, kernel, padding=(r, 0))

        # remove batch and channel dimensions and return CROP_SIZE x CROP_SIZE filtered image
        return filtered.squeeze(0).squeeze(0)
    
    def verticalBoxFilter(self, L, r=5):
        # normalization factor
        weights = torch.ones((L.shape[0], 1), device=L.device, dtype=L.dtype)
        Nv = self.verticalBoxFilterUnnormalized(weights, r)
        
        # return normalized box filter result
        return self.verticalBoxFilterUnnormalized(L, r) / Nv
    
    def get_Sigma(self, I):
        """ Computes a robust estimate of standard deviation (scale) based on the Median Absolute Deviation (MAD) """
        # compute the difference along the horizontal direction I: HEIGHT x WIDTH
        dx = torch.diff(I, dim=1)

        # flatten dx and compute the median of the absolute differences from the median
        dx_median = torch.median(dx)
        mad = torch.median(torch.abs(dx - dx_median))

        # constant factor for normalizing MAD to standard deviation (assuming Gaussian distribution)
        return 1.4826 * mad