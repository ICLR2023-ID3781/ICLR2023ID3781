import numpy as np
import torch
import torch.nn as nn


''' DC '''
def DC(H_A, H_B):
    dis_matA = torch.sum(H_A * H_A, dim=1, keepdim=True) + torch.sum(H_A * H_A, dim=1,  \
               keepdim=True).T - 2 * torch.matmul(H_A, H_A.T)
    Dis_matA = dis_matA - torch.mean(dis_matA, dim=0, keepdim=True)  \
               - torch.mean(dis_matA, dim=1, keepdim=True) + torch.mean(dis_matA) 
    dis_matB = torch.sum(H_B * H_B, dim=1, keepdim=True) + torch.sum(H_B * H_B, dim=1,  \
               keepdim=True).T - 2 * torch.matmul(H_B, H_B.T)
    Dis_matB = dis_matB - torch.mean(dis_matB, dim=0, keepdim=True)  \
               - torch.mean(dis_matB, dim=1, keepdim=True) + torch.mean(dis_matB) 

    numer = torch.mean(Dis_matA * Dis_matB)
    denom = torch.sqrt(torch.mean(Dis_matA * Dis_matA) * torch.mean(Dis_matB * Dis_matB))
    DC_coe = numer / denom
    return DC_coe

''' RV '''
def RV(H_A, H_B):
    K_A = H_A @ H_A.T
    K_B = H_B @ H_B.T
    numerator = torch.trace(K_A @ K_B)   
    denominator = torch.sqrt(torch.trace(K_A @ K_A) * torch.trace(K_B @ K_B))
    RV_coe = numerator / denominator
    return RV_coe

''' MC '''
def MC(H_A, H_B):
    numerator = torch.trace(H_A.T @ H_B)   
    denominator = torch.sqrt(torch.trace(H_A.T @ H_A) * torch.trace(H_B.T @ H_B))
    MC_coe = numerator / denominator
    return MC_coe


''' HSIC '''
# linear kernel function
def linear_kernel(H_A):
    return torch.mm(H_A, H_A.T)

# gaussian kernel function
def gaussian_kernel(H_A, sigma):
    # sigma: width parameter
    dis_mat = torch.sum(H_A * H_A, dim=1, keepdim=True) + torch.sum(H_A * H_A, dim=1,  \
              keepdim=True).T - 2 * torch.matmul(H_A, H_A.T)
    return torch.exp(- dis_mat / sigma)

def hsic1(H_A, H_B):
    # utilize linear kernel
    K_A, K_B = linear_kernel(H_A), linear_kernel(H_B)
    # utilize linear kernel
    # K_A, K_B = gaussian_kernel(H_A, sigma), gaussian_kernel(H_B, sigma)
    N = H_A.shape[0]
    J = torch.eye(N) - torch.ones(N, N) / N
    return torch.trace(K_A @ J @ K_B @ J) / (N  - 1) ** 2

def hsic2(H_A, H_B):
    # utilize linear kernel
    K_A, K_B = linear_kernel(H_A), linear_kernel(H_B)
    # utilize linear kernel
    # K_A, K_B = gaussian_kernel(H_A, sigma), gaussian_kernel(H_B, sigma)
    N = H_A.shape[0]
    K_AB = torch.mm(K_A, K_B)
    hsic_ceo = torch.trace(K_AB) + torch.mean(K_A) * torch.mean(K_B) * N ** 2  \
           - 2 * torch.mean(K_AB) * N
    return hsic_ceo / (N - 1) ** 2
    
def HSIC(H_A, H_B):
    # normalize HSIC
    numerator = hsic2(H_A, H_B)
    denominator = torch.sqrt(hsic2(H_A, H_A) * hsic2(H_B, H_B))
    return numerator / denominator


if __name__ == "__main__":
    ha = torch.randn(128, 32)
    hb = torch.randn(128, 32)
    hsic_val = RV(ha, ha)
    print(hsic_val)



