import numpy as np
from numpy.random import poisson
import scipy
from scipy.stats import poisson
from scipy.stats import norm
import matplotlib.pyplot as plt
from gp import *

class es_sampler():

    def __init__(self, mean, cov,log_L):
        #self.gp_prior = gp_prior
        self.log_L = log_L
        self.mean = mean
        self.cov = cov


    def es(self,f_current):
        """
        Input:
        f_current: current state f (vector of latent variables we wish to sample)
        """

        # choose ellipse - gp prior samples
        mean = np.zeros(self.mean.shape)
        v = np.random.multivariate_normal(mean, self.cov)
        # set log-likelihood threshold
        u = np.random.uniform()
        log_y = self.log_L(f_current) + np.log(u)

        # set barcket for angle variable theta
        theta = np.random.uniform(0,2.*np.pi) # initial proposal
        theta_min, theta_max = theta-2.*np.pi, theta

        while True:
            f_prime = f_current*np.cos(theta)+v*np.sin(theta)
            logl_prime = self.log_L(f_prime)
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
        samples = np.zeros((n_samples, self.cov.shape[0]))
        print(samples.shape)
        samples[0] = np.random.multivariate_normal(mean=self.mean, cov=self.cov)
        for i in range(1,n_samples):
            samples[i] = self.es(samples[i-1])

        return samples



def main():
    np.random.seed(1234)

    mu_1, mu_2 = 5., 1.
    sigma_1, sigma_2 = 1., 2.
    mu = ((sigma_1**-2)*mu_1 + (sigma_2**-2)*mu_2) / (sigma_1**-2 + sigma_2**-2)
    sigma = np.sqrt((sigma_1**2 * sigma_2**2)/(sigma_1**2 + sigma_2**2))


    def log_L(f):
        return norm.logpdf(f,mu_2,sigma_2)

    sampler = es_sampler(np.array([mu_1]),np.diag(np.array([sigma_1**2,])),log_L)
    samples = sampler.sample(n_samples = 500)

    r = np.linspace(0.,8.,num=100)
    plt.figure(figsize=(17,6))
    plt.hist(samples,bins=30,normed=True)
    plt.plot(r,norm.pdf(r,mu,sigma))
    plt.grid()
    plt.show()



if __name__ == "__main__":
    main()
