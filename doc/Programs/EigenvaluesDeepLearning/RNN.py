import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from hamiltonian import *

class EigRNN:
    """
    Finds the lowest eigenvalue of a symmetric matrix
    by minimizing the Rayleigh quotient with gradient
    descent
    """
    def __init__(self,A,eps=1e-4,maxiter=1000):
        """
        A - input matrix
        eps - stops iterating when eigenvalue is not changing more than eps
        maxiter - maximum number of iterations
        """
        self.A = A
        self.maxiter = maxiter
        self.eps = eps
        self.x = np.random.rand(self.A.shape[0])

    def dR(self):
        """
        Returns the gradient of the Rayleigh quotient
        """
        x = self.x
        self.Ax = self.A@x
        self.xTx = x.T@x
        self.eig = x.T@self.Ax/self.xTx
        self.grad = 2*((self.xTx)*self.Ax - (x.T@self.Ax)*x)/((self.xTx)*(self.xTx))
        return(self.grad)

    def optStep(self):
        """
        Returns the optimal step length
        """
        dR = self.grad
        x = self.x
        xTA = self.Ax.T
        a = xTA@x
        b = xTA@dR
        c = dR.T@self.A@dR
        e = self.xTx
        f = x.T@dR
        g = dR.T@dR
        return( ( (a*g - e*c ) + np.sqrt( (e*c - a*g)**2 - 4*(a*f - e*b)*(b*g - c*f) ) )/(2*(b*g - c*f)))

    def solve(self):
        """
        Returns the lowest eigenvalue of A.
        To be called after initialization.
        """
        convergence = False
        val = 0
        for i in range(self.maxiter):
            grad = self.dR()
            self.x = self.x - self.optStep()*grad
            if np.sum(np.abs(grad)) < self.eps*(np.log(self.x.shape[0])):
                convergence = True
                break

        if not convergence:
            print('WARNING: Did not converge. Try increasing maxiter or a smaller eps.')    

        return(self.eig,self.x)



if __name__ == '__main__':
    n_pairs = int(sys.argv[1])
    n_basis = int(sys.argv[2])
    print('System with {} pairs and {} basis states'.format(n_pairs,n_basis))
    delta = float(sys.argv[3])
    g = float(sys.argv[4])
    epsilon = float(sys.argv[5])
    H,Eref = hamiltonian(n_pairs,n_basis,delta,g)
    RNN = EigRNN(H,eps=epsilon)
    eigval,eigvec = RNN.solve()
    print('Energy: {}'.format(eigval))
