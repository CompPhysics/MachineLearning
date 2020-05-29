import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

"""
Download https://physics.bu.edu/~pankajm/ML-Review-Datasets/isingMC/IsingData.zip
Unzip files in the same folder as this script.
Run script
"""

def read_t(t=0.25,root="./"):
    if t > 0.:
        data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=%.2f.pkl'%t,'rb'))
    return np.unpackbits(data).astype(int).reshape(-1,1600)

stack = []
for i,t in enumerate(np.arange(0.25,4.01,0.25)):
    y = np.ones(10000,dtype=int)
    if t > 2.25:
        y*=0
    stack.extend(list(y))

y = np.array(stack)

stack = []
for t in np.arange(0.25,4.01,0.25):
    stack.append(read_t(t))

X = np.vstack(stack)

