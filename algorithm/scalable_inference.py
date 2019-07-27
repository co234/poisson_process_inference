import numpy as np
from numpy.random import poisson
import scipy
from scipy.stats import poisson
import matplotlib.pyplot as plt
from gp import *
from scipy.stats import bernoulli
import random
import math
from Elliptical_Slice import *

a = 0.
b = 50.
s = b-a
k = np.linspace(a,b,20)
#theta = [1,18,0.002]
theta = [0.5,19,0.002]
s_f = theta[0]
l = theta[1]
s_n = theta[2]
#G = generate_gp_sample(se_kernel(k,k,s_f,l),k)


def f1(s):
    return 2*np.exp(-s/15) + np.exp(-((s-25)/10)**2)

x = np.linspace(0,50,100)
plt.plot(x,f1(x))
#plt.xlim(0,50)
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

X=np.linspace(0,50,100)
y = [0]*len(events)
plt.plot(X,f1(X))
plt.plot(events,y,'r+')
plt.show()
print(len(events))

sampler = es_sampler(k,0.,50.,20,events,theta)

import time as time
t0 = time.time()
loglamda = sampler.sample(50)
t1 = time.time()

print('Runtime: %.3f mins' % ((t1-t0)/60))

test_log = loglamda.mean(axis=0)
plt.plot(k,np.exp(test_log))
plt.show()

mean_total = []
for i in loglamda:
    X_test = np.linspace(0,50,100)
    #X_test = events
    #mean_pre, cov_pre = gp_inference(k,i,X_test,[0.05,10,0.002])
    mean_pre, cov_pre = gp_inference(k,i,X_test,theta)
    mean_pre_ = mean_pre + np.diag(cov_pre)
    mean_total.append(mean_pre)
mean = np.array(mean_total).mean(axis=0)

X=np.linspace(0,50,1000)
y = [0.1]*len(events)
plt.plot(X,f1(X))
plt.plot(events,y,'r+')
plt.plot(X_test,np.exp(mean))
plt.xlim(0,50)
plt.ylim(0,4)
plt.show()
