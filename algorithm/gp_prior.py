import numpy as np
from numpy.random import poisson
import scipy
from scipy.stats import poisson
import matplotlib.pyplot as plt
from gp import *
from scipy.stats import bernoulli
import random
import math

class sampler():
    def __init__(self, t1, t2, theta,upper_bound):
        self.t1 = t1
        self.t2 = t2
        self.T = t2-t1
        self.V = self.T
        self.upper_bound = upper_bound
        self.theta = theta
        self.s_f = theta[0]
        self.l = theta[1]
        self.s_n = theta[2]

    def gp_function(self):
        np.random.seed(1234)
        V = self.T     # measure of T - interval
        J = poisson.rvs(self.upper_bound*V, random_state=1234)
        X = np.linspace(self.t1, self.t2,1000)[:,np.newaxis]
        cov = se_kernel(X,X,self.s_f, self.l)
        gp_set = np.array(generate_gp_sample(cov,X)).T
        Sj_hat = np.sort(np.random.uniform(self.t1,self.t2,J))

        bin_size = self.T/1000
        idx = np.round(Sj_hat/bin_size).astype(int)
        g_sample = [gp_set[i] for i in idx]
        sig_g_sample = [self.sigmoid(g) for g in g_sample]

        Sj = []
        g_Sj = []
        for i in range(J):
            r = np.random.uniform()
            if r < sig_g_sample[i]:
                Sj.append(Sj_hat[i])
                g_Sj.append(g_sample[i])

        plt.plot(X,self.sigmoid(gp_set)*self.upper_bound)
        y = [0.5]*len(Sj)
        y_upper = [self.upper_bound]*len(Sj)

        plt.plot(Sj,y,'r+')
        plt.plot(Sj,y_upper)
        plt.ylim(0, self.upper_bound+1)
        plt.xlim(0, Sj[-1])
        plt.show()

        return Sj,list(g_Sj)


    def SGCP(self):
        """
        output: 1-d array
        """
        np.random.seed(1234)

        self.V = self.T     # measure of T - interval
        self.J = poisson.rvs(self.upper_bound*self.V, random_state=1234) # generate rv from poisson dist.

        Sj = np.sort(np.random.uniform(self.t1,self.t2,self.J))[:,np.newaxis]

        # generate gp samples
        G = np.array(generate_gp_sample(se_kernel(Sj,Sj,self.s_f,self.l),Sj))

        # sigmoid transform
        G_sigmoid = self.trans_sigmoid(G).T
        #print("G_sigmoid's shape: ", G_sigmoid.shape)
        events = []
        gp_events = []
        sig_gp = []
        gp_points = []

        # params for plotting
        for j in range(self.J):
            r = np.random.uniform()
            if r < G_sigmoid[j]:
                events.append(Sj[j])
                gp_events.append(G[j])
                sig_gp.append(G_sigmoid[j])
                gp_points.append(G_sigmoid[j]*self.upper_bound)

        plt.plot(events,gp_points)
        y = [0.5]*len(events)
        y_upper = [self.upper_bound]*len(events)

        plt.plot(events,y,'r+')
        plt.plot(events,y_upper)
        plt.ylim(0, self.upper_bound+1)
        plt.xlim(0, self.T)
        plt.show()

        return events, list(gp_events)

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def function_value_sampler(self,x,y):
        for idx in range(len(x)):
            a = x[idx]
            x_temp = x[:idx] + x[idx+1:]
            y_temp = y[:idx] + y[idx+1:]
            a_mean, a_cov = gp_inference(x_temp,y_temp,a,self.theta)
            new_y = np.asscalar(np.random.normal(a_mean, np.sqrt(a_cov)))
            y[idx] = new_y

        return y


    def M_sampler(self, observation, g_observation, burnin = 10, niter = 0):

        # Initialisation
        total_iter = niter + burnin
        M_global = []
        sm_global = []
        g_sm_global = []
        g_k_global = []
        # Start with no latent rejections
        M = 0
        g_sm=[]
        sm = []

        # Initialise the function at the data
        sk, g_k = observation, g_observation
        for r in range(total_iter):
            b_rv = bernoulli.rvs(0.5) # generate bernoulli random variable 0 or 1
            if b_rv == 0:
                # Add thinned points
                s_pos = np.random.uniform(self.t1,self.t2)
                # draw corresponding function value from GP condition on current state gm+k
                s_pos_mean, s_pos_cov = gp_inference(list(sm)+list(sk),\
                                        list(g_sm)+list(g_k),s_pos,self.theta)
                g_s_pos = np.asscalar(np.random.normal(s_pos_mean,np.sqrt(s_pos_cov)))

                # Compute accpetence ratio
                a_ins = (self.V*self.upper_bound)/((M+1)*(1+np.exp(g_s_pos)))
                u_ins = np.random.uniform()
                if u_ins < a_ins:
                    sm.append(s_pos)
                    g_sm.append(g_s_pos)
                    M+=1

            elif M>0:
                # selcet one of the thinned point randomly
                m = math.ceil(np.random.uniform(0,M))
                xm = sm[m-1]
                g_s_cur = g_sm[sm.index(xm)]
                a_del = (M*(1+np.exp(g_s_cur)))/(self.V*self.upper_bound)
                u_del = np.random.uniform()
                if u_del < a_del:
                    sm.remove(xm)
                    g_sm.remove(g_s_cur)
                    M-=1

            for mi in range(M):
                # Sample from Truncated normal distribution
                while True:
                    x = np.random.normal(sm[mi],0.5)
                    if x >= self.t1 and x<= self.t2:
                        break
                s_hat = float(x)

                s_hat_mean, s_hat_cov = gp_inference(list(sm)+list(sk),list(g_sm)+list(g_k),s_hat,self.theta)
                g_s_hat = np.asscalar(np.random.normal(s_hat_mean,np.sqrt(s_hat_cov)))
                a_loc = (1+np.exp(g_sm[mi]))/(1+np.exp(g_s_hat))
                u_loc = np.random.uniform()

                # assert
                if u_loc < a_loc:
                    sm[mi] = s_hat
                    g_sm[mi] = g_s_hat

            # update gm and gk using gibbs sampling
            all_events = list(sm)+list(sk)
            all_gp = list(g_sm) + list(g_k)
            g_sm_k = self.function_value_sampler(all_events,all_gp)
            g_sm =  g_sm_k[:-len(sk)]
            g_k = g_sm_k[-len(sk):]
            if r%50==0:
                print("Iteration: ",r)
            # Append local variables to gloabl variable
            M_global.append(M)
            sm_global.append(list(sm))
            g_sm_global.append(list(g_sm))
            g_k_global.append(list(g_k))

        return sk, M_global,sm_global,g_sm_global,g_k_global


def main():
    pass

if __name__ == "__main__":
    main()
