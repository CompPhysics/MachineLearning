from pylab import*
from datetime import datetime
import numpy as np
from scipy.sparse import diags
from numpy import linalg as LA
import unittest 


#Fuction to find the values of cosinus and sinus
def Rotation(A,R,k,l,n):
    if (A[k,l] !=0):
        tau = (A[l,l] - A[k,k])/float(2*A[k,l])
        if (tau > 0): 
            t = 1/(tau + sqrt(1 + tau*tau))
        else:
        	t = -1/(-tau + sqrt(1 + tau*tau))
        
        c = 1/float(sqrt(1+t*t))
        s = c*t
        
    else: 
        c = 1
        s = 0
    

    a_kk = A[k,k]
    a_ll =A[l,l]
    
    #Changing the matrix elements with indices k and l
    A[k,k]= c**(2)*a_kk -2*c*s*A[k,l] +s**(2)*a_ll
    A[l,l] = s**(2)*a_kk + 2*c*s*A[k,l] + c**(2)*a_ll
    
    A[k,l] = 0
    A[l,k] = 0
    

    for i in range(0,n+1):
        if (i != k and i != l):
            a_ik =A[i,k] #definerer
            a_il= A[i,l] #definerer 

            A[i,k]= c*a_ik -s*a_il
            A[k,i]= A[i,k]

            A[i,l]= c*a_il + s*a_ik
            A[l,i]= A[i,l]   
          
     #Finner de nye egenvektorene 
	r_ik =R[i,k]
	r_il= R[i,l]

	R[i,k]= c*r_ik -s*r_il
	R[i,l] = c*r_il +s*r_ik

	#print "A_ROTA:", A
	#print "R_ROTA:", R

	return k,l, A,R
	 


 #Fuction to find maximum matrix element.
def MaximumOffDiagonal(n,A):

	maxx = 0 
	k,l=0,0
	for i in range (1,n+1):
		for j in range (1,n+1):
			if (abs(A[i,j]) > epsilon): 
				maxx= abs(A[i,j])
				k= i
				l =j

	return maxx, k,l 



def JacobiMethod(A,R,n):	
	for i in range (1,n+1):
		for j in range (1,n+1):
			if (i==j):
				R[i,j]=1
			else:
				R[i,j]=0

	
	epsilon = 10**(-8)
	max_number_iterations = n**(3)
	iterations=0
	
	maksoff_diagonal,k,l= MaximumOffDiagonal(n, A)

	while ((abs(maksoff_diagonal) > epsilon) and (iterations < max_number_iterations) ):
		maksoffdiagonal,k,l = MaximumOffDiagonal(n,A)
		eigenvalue= Rotation(A,R,k,l,n)
		iterations  += 1


	print "antall iterasjoner=",  iterations 

		
	#print "egenverdier:", eigen_valuess #halllooooo
	return A,R,n,iterations


n = 2
rho_min=0 
rho_max = 5
h= (rho_max - rho_min)/float(n+1)

rho = zeros(n+1)
V= zeros(n+1)
d= zeros(n+1)
e= zeros(n+1)

e = -1/float(h**2)

A= zeros((n,n))
R= zeros((n,n))

for i in range(0,n+1):
	rho[i]=rho_min +i*h
	V[i]=rho[i]**2
	d[i]= 2/float(h**2) + V[i]

o = np.array([e*np.ones(n),d*np.ones(n+1),e*np.ones(n)])
offset = [-1,0,1]

A= diags(o,offset).toarray()
#print "A"

print "A=",A

s= np.array([0*np.ones(n),np.ones(n+1),0*np.ones(n)])
R = diags(s,offset).toarray()
print  "R= ", R


epsilon= 1*10**(-8)


MaximumOffDiagonal(n,A)

tstart= datetime.now()
print tstart
A_jacobi,R,n,iterations = JacobiMethod(A,R,n)
tend= datetime.now()
print tend
print "Endring i tid:", tend- tstart

print "A_diag=", A
print "A_jacobi", A_jacobi
print "R_nyyy",R

eig_vals,eig_vecs =  np.linalg.eig(A) #egenverdi, egenvektor 

eig_vals_sorted = np.sort(eig_vals)
eig_vecs_sorted = eig_vecs[:,eig_vals.argsort()]
print "egenverdi_sortet=", eig_vals_sorted
print "egenvektor_sortet=", eig_vecs_sorted


#print np.sort(np.linalg.eig(A[:,j]))


#Rotation(A,R,k,l,n) #jacobi egenverdiene


#print "antall iterasjoner=",  iterations 
#print "egenverdier:", eigen_valuess 



#eigenvalues
#TIDEN til EIGENVALUES

# tstart= datetime.now()
# print tstart

# for i in range(1,6):
# 	eig_vals[i] = np.linalg.eig(A[i,i])
# eig_vals_sorted = np.sort(eig_vals)  #Sorting eigenvalues from the lowest to the biggest

# print "eigvals_numpy=", eig_vals_sorted

# tend= datetime.now()
# print tend

# print "Endring i tid:", tend- tstart 





