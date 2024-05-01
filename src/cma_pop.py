import numpy as np
from scipy.linalg import fractional_matrix_power
from plotting import draw_contour
import sys
try:
    sys.path.append('./')
    from cec2017.functions import *
except Exception:
    sys.path.append('../')
    from cec2017.functions import *

np.set_printoptions(3)

class CMAES():
    # equal weights, then mu_eff=mu

    #todo cs
    def __init__(self, l: int=20, mu: int=None, sigma: float=0.5, cs: float=None, cc: float=None, ds = None) -> None:
        # initialises solver
        # l - (lambda) population size
        # mu - selection size for new mean calculation, by defaulf lambda/2
        self.l = l
        self.mu = mu if mu else l//2
        self.sigma = sigma; # step size, depends on problem
        self.n = 2 # problem dimensionality, hardcoded to 2 for this project

        # exponential decay params and stuff from book
        self.cs = cs if cs else (self.mu+2)/(2+self.mu+5)
        self.cc = cc if cc else (4+self.mu/2)/(6+self.mu)
        self.c1 = 2/(3.3*3.3+self.mu)
        self.ds = 2.21#1+2*max(0,np.sqrt((self.mu-1)/3)-1)+self.cs
        self.cmu = min(1-self.c1,2*(self.mu-2+1/self.mu)/(16+self.mu))

        self.generation=0
    
    def update_state(self,d):
        # mean
        rank_mu = sum(np.array([(self.mean-di)/self.l]).T@np.array([(self.mean-di)/self.l]) for di in d) # for later
        delta = np.average(d-self.mean,(0))/self.sigma #!!! all weights are 1/mu
        self.mean+= self.sigma*delta
        #print("mean delta,new mean:",delta,self.mean)

        # cov matrix and step size path
        # the idea is that instead of just overwriting pc with new delta, exponential smoothing is used
        # the sqrt is normalization factor something because we want pc ~ N(0,C)

        # hsig = (sum(sum(x**2) for x in self.ps) / 2  # ||ps||^2 / N is 1 in expectation
        #         / (1-(1-self.cs)**(2*self.generation/self.l))  # account for initial value of ps
        #         < 2 + 4./(2+1))  # should be smaller than 2 + ...
        
        self.pc = (1-self.cc)*self.pc+np.sqrt(self.cc*(2-self.cc)*self.mu)*delta

        # here it's the same, but something something eigenvalue decomposition of C,
        # because the expected length of ps depends on it's rotation, then C^-1/2 is BD^2B^T
        # where D^2 are eigenvalues and B is basis for eigenvectors
        normalisation = np.sqrt(self.cs*(2-self.cs)*self.mu)/self.sigma
        self.ps = (1-self.cs)*self.ps+normalisation*fractional_matrix_power(self.C,-0.5)@delta

        #print("ps",self.ps,"normalisation",normalisation,"matrix",fractional_matrix_power(self.C,-0.5),"delta",delta)

        # E||N(0,I)||= sqrt(n)=sqrt(2) for this project
        self.sigma = self.sigma*np.exp((self.cs/self.ds)*(np.linalg.norm(self.ps)/np.sqrt(2)-1))
        #print("step size:",self.sigma)

        # C must be calculated after pc is updated, but with old mean
        #cmu*2 magic multiply by 2 because purecma says so
        self.C = (1-self.c1-self.cmu*2)*self.C+self.c1*(self.pc.T@self.pc+self.C)+self.cmu*rank_mu/self.sigma**2
        print("cov matrix:",self.C)

    def step(self,f):
        self.generation+=self.l
        # if self.generation==6:
        #   d = np.array([[19.70717688313625, 17.892611615608576], [20.27093664976886, 18.79446561569581], [19.94685865736256, 19.70977572937749], [19.81613177890674, 20.03250412907758], [21.413507790048122, 18.467364128909118], [20.698827693253982, 19.903632154984724]])      
        # else:
        d = np.random.multivariate_normal(self.mean,self.sigma*self.sigma*self.C,self.l)
        draw_contour(f, d,(-100,100),(-100,100),self.mu)
        d_sorted = self.get_n_best_by_target(d,f,self.mu)
        self.update_state(d_sorted)

    def get_n_best_by_target(self,population, target, n):
        q = [target(p) for p in population]
        permute = np.argsort(q)[:n] # get n best
        return population[permute]


    def run(self, f, mean=None, max_iterations=100):
        # minimises function f

        # reset state
        self.mean= mean if mean else np.zeros(self.n)
        self.C = np.eye(self.n)
        self.pc = np.zeros((1,self.n)) # covariance evolution path
        self.ps = np.zeros((1,self.n)) # sigma (step) path

        for g in range(max_iterations):
            self.step(f)

if __name__=="__main__":
    target = lambda X: np.dot(X,X) # x^2+y^2, but for any dimension
    solver = CMAES(l=50)
    solver.run(f4,mean=[-15,-15])