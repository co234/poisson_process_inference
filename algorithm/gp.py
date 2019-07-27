import numpy as np
from numpy.random import poisson
import scipy
from scipy.stats import poisson
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

def se_kernel(xi,xj,s_f,l):
    #np.random.seed(1234)
    """
    the kernel is sysmetric
    hyperparameter:
    1. s_f: signal variance - determine variation of function values from mean
            small -> close to mean
            large -> more variation
    2. l: lengthscale - determine how smooth the function is
          small value -> change quickly
          large value -> change slowly
    - aussume noise vairance is 0 for now(can be added later)
    """
    if isinstance(xi,float) == True:
        k = np.exp(-(xi-xj)**2/(2*l**2))*s_f**2

    elif isinstance(xj,float) == True:
        xj_arr = np.array([xj])
        m = len(xi) # dim of xi
        n = len(xj_arr) # dim of xj
        k = np.zeros((m,n)) # initialise kernel
        for i in range(m):
            for j in range(n):
                k[i,j] = np.exp(-(xi[i]-xj_arr[j])**2/(2*l**2))*s_f**2
    else:
        m = len(xi) # dim of xi
        n = len(xj) # dim of xj
        k = np.zeros((m,n)) # initialise kernel
        for i in range(m):
            for j in range(n):
                k[i,j] = np.exp(-(xi[i]-xj[j])**2/(2*l**2))*s_f**2

    return k

def generate_gp_sample(kernel,x_i):
    """
    input:
    1. number of samples
    2. kernel - 2d array
    3. data - 1d array

    output:
    samples from gp prior - 1d array if sample = 1
    """
    m = kernel.shape[0]
    np.random.seed(1234)
    #  cholesky decompositon: L = cholesky(K+s_n^2I)
    L = np.linalg.cholesky(kernel+np.eye(m)*1e-10)

    x_ = np.matmul(L,np.random.normal(0,1,m))
    #plt.plot(x_i,x_)
    #plt.show()

    return x_



def gp_inference(x,t,x_new,theta):
    #np.random.seed(1234)
    s_f,l,s_n = theta[0],theta[1],theta[2]

    k_x,k_xt,k_xtxt = se_kernel(x,x,s_f,l), se_kernel(x,x_new,s_f,l),\
                        se_kernel(x_new,x_new,s_f,l)

    k_inv = np.linalg.pinv(k_x+(s_n**2)*np.eye(k_x.shape[0]))

    xt_mean = np.matmul(np.matmul(k_xt.T,k_inv),t)
    xt_cov = k_xtxt - np.matmul(np.matmul(k_xt.T,k_inv),k_xt)
    # Add small value to the diagonal elements aviod overflow
    #xt_cov_flow = np.eye(xt_cov.shpae[0]) + xt_cov

    return(xt_mean,xt_cov)
