import scipy as scipy
import scipy.special as special
import numpy as np
import itertools as it
from pandas import *
import matplotlib.pylab as plt
from scipy.integrate import ode
import time

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def hamiltonian(n_pairs,n_basis,delta,g):
	"""
	n_pairs - Number of electron pairs
	n_basis - Number of spacial basis states
	"""
	n_SD = int(special.binom(n_basis,n_pairs))
	print("n = ", n_SD)
	H_mat = np.zeros((n_SD,n_SD))
	S = stateMatrix(n_pairs,n_basis)
	for row in range(n_SD):
		bra = S[row,:]
		for col in range(n_SD):
			ket = S[col,:]
			if np.sum(np.equal(bra,ket)) == bra.shape:
				H_mat[row,col] += 2*delta*np.sum(bra - 1) - 0.5*g*n_pairs
			if n_pairs - np.intersect1d(bra,ket).shape[0] == 1:
				H_mat[row,col] += -0.5*g
	return(H_mat)


def stateMatrix(n_pairs,n_basis):
	L = []
	states = range(1,n_basis+1)
	for perm in it.permutations(states,n_pairs):
		L.append(perm)
	L = np.array(L)
	L.sort(axis=1)
	L = unique_rows(L)
	return(L)



g = 0.5

H = hamiltonian(4,8,1,g)
print("Hamiltonian calculated")

A = H
x = np.zeros(A.shape[0])
x[0] = 1

def f(t,x):
	return(-(x.T@x)*A@x + (x.T@A@x)*x)



r = ode(f)
r.set_initial_value(x,0) #langsos algorithm

t1=10
dt=0.1
start = time.time()
while r.successful() and r.t < t1:
	r.integrate(r.t+dt)
end = time.time()



print("RNN eig: ",r.y.T@A@r.y/(r.y.T@r.y))
print("calculation time with RNN: ", end - start)


start = time.time()
eigvals, eigvecs = np.linalg.eig(H)
end = time.time()
print("numpy eig: ",np.sort(eigvals))
print("calculation time with numpy eig: ", end - start)
