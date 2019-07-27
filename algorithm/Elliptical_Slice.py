import numpy as np
from numpy.random import poisson
import scipy
from scipy.stats import poisson
import matplotlib.pyplot as plt
from gp import *


class es_sampler():

    def __init__(self,k,a,b,p,obs,theta):
        self.k = k
        self.a = a
        self.b = b
        self.p = p
        self.obs = obs
        self.theta = theta
        self.k_x= se_kernel(self.k,self.k,self.theta[0],self.theta[1])
        self.k_inv = np.linalg.pinv(self.k_x+(self.theta[2]**2)*np.eye(self.k_x.shape[0]))

    def m(self,s,x,G):
        k_xt = se_kernel(x,s,self.theta[0],self.theta[1])
        mean_s = np.matmul(np.matmul(k_xt.T, self.k_inv),G)

        return np.asscalar(mean_s)

    def covariance(self,s1,s2,x):
        k_xt, k_xtxt = se_kernel(x,s1,self.theta[0], self.theta[1]), se_kernel(s1,s2,self.theta[0],self.theta[1])
        cov_s = k_xtxt - np.matmul(np.matmul(k_xt.T,self.k_inv),k_xt)

        return np.asscalar(cov_s)

    def f(self,s,G):
        ms = self.m(s,self.k,G)
        gamma_s = self.covariance(s,s,self.k)

        return np.exp(ms+0.5*gamma_s)

    def g(self,s1,s2,G):
        '''
        g(s1,s2) = exp(m(s1)+m(s2)+gamma(s1,s2)+0.5gamma(s1,s1)+0.5gamma(s2,s2))
        '''
        ms_1, ms_2 = self.m(s1,self.k,G), self.m(s2,self.k,G)
        gamma_s1,gamma_s12,gamma_s2 = self.covariance(s1,s1,self.k),\
                                      self.covariance(s1,s2,self.k),\
                                      self.covariance(s2,s2,self.k)

        return np.exp(ms_1+ms_2+gamma_s12+0.5*gamma_s1+0.5*gamma_s2)

    def mu_integral(self,G):
        X, W = np.polynomial.legendre.leggauss(self.p)  # np.ndarray
        S = (self.b-self.a)*0.5*X+(self.b+self.a)*0.5 # S is a vector
        y = np.array([self.f(i,G) for i in S])
        #print(y)
        return (self.b-self.a)*0.5*np.dot(W,y)

    def cov_integral(self,G):
        X, W = np.polynomial.legendre.leggauss(self.p)
        xi = (self.b-self.a)*0.5*X+(self.b+self.a)*0.5
        xj = (self.b-self.a)*0.5*X+(self.b+self.a)*0.5
        #g_ = np.array([g(xi[i],xj[i],k,G) for i in range(len(xi))])

        mu = self.mu_integral(G)
        cov_I = 0
        for j in range(self.p):
            for i in range(self.p):
                #cov_I_ = W[i]*W[j]*g_[i]
                cov_I_ = W[i]*W[j]*self.g(xi[i],xj[j],G)
                cov_I += cov_I_
        #print(((b-a)**2)*0.25*np.dot(W*W,g_))
        #print(mu**2)
        #print(mu)
        assert cov_I > 0

        return cov_I


    def gamma_params(self,G):
        mean = self.mu_integral(G)
        cov = self.cov_integral(G)
        alpha_ = mean**2/cov
        beta_ = cov/mean

        return alpha_,beta_



    def log_Lik(self,f):
        alpha, beta = self.gamma_params(f)
        mean, cov = gp_inference(self.k,f,self.obs,self.theta)
        lik = np.exp(mean.sum() + 0.5*np.trace(cov))*(1+beta)**(-alpha)

        return np.log(lik)

    def es(self, f_current):
        """
        Input:
        f_current: current state f (vector of latent variables we wish to sample)
        """
        # choose ellipse - gp prior samples
        #kernel = se_kernel(self.k,self.k,self.theta[0],self.theta[1])
        v = generate_gp_sample(self.k_x,self.k)

        # set log-likelihood threshold
        u = np.random.uniform()
        log_y = self.log_Lik(f_current) + np.log(u)

        # set barcket for angle variable theta
        theta = np.random.uniform(0,2.*np.pi) # initial proposal
        theta_min, theta_max = theta-2.*np.pi, theta

        while True:
            f_prime = f_current*np.cos(theta)+v*np.sin(theta)
            logl_prime = self.log_Lik(f_prime)
            if logl_prime > log_y:
                break
            else:
                if theta < 0:
                    theta_min = theta
                else:
                    theta_max = theta

                theta = np.random.uniform(theta_min, theta_max)

        return f_prime

    def sample(self, n_samples=1000):

        #kernel = se_kernel(self.k,self.k,self.theta[0],self.theta[1])
        samples = np.zeros((n_samples, self.k_x.shape[0]))
        #print(samples.shape)
        #samples[0] = np.random.multivariate_normal(mean=self.mean, cov=self.cov)

        samples[0] = generate_gp_sample(self.k_x,self.k)
        for i in range(1,n_samples):
            samples[i] = self.es(samples[i-1])

        #sample = generate_gp_sample(self.cov,k)
        return samples
