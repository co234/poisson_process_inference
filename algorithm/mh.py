import numpy as np
from numpy.random import poisson
import scipy
from scipy.stats import poisson
import matplotlib.pyplot as plt
from gp import *


k_x= se_kernel(k,k,s_f,l)
k_inv = np.linalg.pinv(k_x+(s_n**2)*np.eye(k_x.shape[0]))


def m(s,x,G):
    '''
    Input params:
    s: s1 for every point in vector S
    x: inducing points k
    '''
    #s_f,l,s_n = theta[0],theta[1],theta[2]
    #k_x,k_xt = se_kernel(x,x,s_f,l),se_kernel(x,s,s_f,l)
    k_xt = se_kernel(x,s,s_f,l)
    #k_inv = np.linalg.pinv(k_x+(s_n**2)*np.eye(k_x.shape[0]))
    #G = generate_gp_sample(k_x,x)
    # m(s) = m* + kernel(s,D')kernel(D'D')^-1G
    #G = generate_gp_sample(se_kernel(k,k,s_f,l),k)
    mean_s = np.matmul(np.matmul(k_xt.T,k_inv),G)

    return np.asscalar(mean_s)

def cov(s1,s2,x):
    #s_f,l,s_n = theta[0],theta[1],theta[2]
    #k_x,k_xt,k_xtxt = se_kernel(x,x,s_f,l),se_kernel(x,s,s_f,l),se_kernel(s,s,s_f,l)
    k_xt,k_xtxt = se_kernel(x,s,s_f,l),se_kernel(s,s,s_f,l)
    #k_inv = np.linalg.pinv(k_x+(s_n**2)*np.eye(k_x.shape[0]))
    cov_s = k_xtxt - np.matmul(np.matmul(k_xt.T,k_inv),k_xt)

    return np.asscalar(cov_s)

def f(s,k,G):
    '''
    f(s) = exp(m(s) + 0.5gamma(s,s))
    '''
    ms = m(s,k,G)
    gamma_s = cov(s,s,k)

    return np.exp(ms+0.5*gamma_s)

def g(s1,s2,k,G):
    '''
    g(s1,s2) = exp(m(s1)+m(s2)+gamma(s1,s2)+0.5gamma(s1,s1)+0.5gamma(s2,s2))
    '''
    ms_1, ms_2 = m(s1,k,G), m(s2,k,G)
    gamma_s1,gamma_s12,gamma_s2 = cov(s1,s1,k),cov(s1,s2,k),cov(s2,s2,k)

    return np.exp(ms_1+ms_2+gamma_s12+0.5*gamma_s1+0.5*gamma_s2)

def mu_integral(a,b,p,k,G):
    X, W = np.polynomial.legendre.leggauss(p)  # np.ndarray
    S = (b-a)*0.5*X+(b+a)*0.5 # S is a vector
    y = np.array([f(i,k,G) for i in S])
    #print(y)
    return (b-a)*0.5*np.dot(W,y)

def cov_integral(a,b,p,k,G):
    X, W = np.polynomial.legendre.leggauss(p)
    xi = (b-a)*0.5*X+(b+a)*0.5
    xj = (b-a)*0.5*X+(b+a)*0.5
    #g_ = np.array([g(xi[i],xj[i],k,G) for i in range(len(xi))])

    mu = mu_integral(a,b,p,k,G)
    ## 2 for loops
    #cov_I = ((b-a)**2)*0.25*np.dot(W*W,g_) - mu**2
    cov_I = 0
    for j in range(p):
        for i in range(p):
            #cov_I_ = W[i]*W[j]*g_[i]
            cov_I_ = W[i]*W[j]*g(xi[i],xj[j],k,G)
            cov_I += cov_I_
    #print(((b-a)**2)*0.25*np.dot(W*W,g_))
    #print(mu**2)
    #print(mu)
    assert cov_I > 0

    return cov_I


def gamma_params(a,b,p,k,G):
    mean = mu_integral(a,b,p,k,G)
    cov = cov_integral(a,b,p,k,G)
    a = mean**2/cov
    b = cov/mean

    return a,b
