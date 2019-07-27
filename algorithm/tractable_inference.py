import numpy as np
from numpy.random import poisson
import scipy
from scipy.stats import poisson
import matplotlib.pyplot as plt
from mcmc import sampler
from gp import *
from scipy.stats import bernoulli
import random
import math
import statistics



def f1(s):
    return 2*np.exp(-s/15) + np.exp(-((s-25)/10)**2)

x = np.linspace(0,50,100)
plt.plot(x,f1(x))
plt.show()

upper_bound,t1,t2 = 2,0,50
#np.random.seed(1234)
J = poisson.rvs(upper_bound*(t2-t1)) # generate rv from poisson dist.
Sj = np.sort(np.random.uniform(t1,t2,J))
G = f1(Sj)

events = []
intensity = []
for j in range(J):
    r = np.random.uniform()
    if r < G[j]:
        events.append(Sj[j])
        intensity.append(G[j])

X=np.linspace(0,50,1000)
y = [0]*len(events)
plt.plot(X,f1(X))
plt.plot(events,y,'r+')
plt.show()

theta = [1,10,0.002]
s_f = theta[0]
l = theta[1]
s_n = theta[2]
V = 50
upper_bound = 2
t1 = 0
t2 = 50

mcmc_sample = sampler(t1,t2,theta,upper_bound)
#obs_test, M_test, thinned_pos_test,gp_thinned_test,gp_obs_test = M_sampler(burnin = 100, niter = 400)
obs_test, M_test, thinned_pos_test,gp_thinned_test,gp_obs_test = mcmc_sample.M_sampler(events,f1(np.array(events)),burnin=100,niter=1)


def sigmoid(z):
    return 1/(1+np.exp(-z))

def extract_position(length):
    # extract samples postition and gp function value
    pos= list(filter(lambda x:x != None, \
                     [thinned_pos_test[i] if x==length else None for i,x in enumerate(M_test)]))

    gp_pos= list(filter(lambda x:x != None, \
                     [gp_thinned_test[i] if x==length else None for i,x in enumerate(M_test)]))

    return pos, gp_pos

def sort_position(obs):
    # Sort by position
    position, gp_value = extract_position(obs)
    for i in range(len(position)):
        x = position[i]
        y = gp_value[i]
        z = list(zip(x,y))

        x_,y_ = zip(*sorted(z,key=lambda x:x[0]))
        position[i] = x_
        gp_value[i] = y_

    return position, gp_value

def mean_value(obs):
    # Get mean value
    x, y = sort_position(obs)
    mean_pos = np.array(x).mean(axis=0)
    mean_gp = np.array(y).mean(axis=0)

    return mean_pos, mean_gp

def plot_thinned_points(x,y,lamda,t):
    # plot thinned points
    sig_y = [sigmoid(i)*lamda for i in y]
    plt.plot(x, sig_y)
    plt.xlim(0,t)
    plt.show()

def plot_thinned_points_s(x,y,lamda,t):
    # scatter plot thinned points
    sig_y = [sigmoid(i)*lamda for i in y]
    plt.scatter(x, sig_y)
    plt.xlim(0,t)
    plt.show()


thinned_points, gp_thinned_points = mean_value(statistics.mode(M_test))
plot_thinned_points(thinned_points,gp_thinned_points,2,50)
plot_thinned_points_s(thinned_points,gp_thinned_points,2,50)
