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
	Returns - The hamiltonian and the reference energy
	"""
	n_SD = int(special.binom(n_basis,n_pairs))
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
	return(H_mat,H_mat[0,0])


def stateMatrix(n_pairs,n_basis):
	L = []
	states = range(1,n_basis+1)
	for perm in it.permutations(states,n_pairs):
		L.append(perm)
	L = np.array(L)
	L.sort(axis=1)
	L = unique_rows(L)
	return(L)

