import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from math import *
import time
# User-friendly machine learning library
# Front end for TensorFlow
import tensorflow.keras
# Different methods from Keras needed to create an RNN
# This is not necessary but it shortened function calls 
# that need to be used in the code.
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Input
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential
#from tensorflow.keras.layers.core import Dense, Activation 
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
# For timing the code
from timeit import default_timer as timer
# For plotting
import matplotlib.pyplot as plt


# Define Analytical, Euler-Cromer, and Velocity-Verlet methods of solving
def analytical(k,m,x0,v0,dt,tfinal):
    t = np.arange(0,tfinal+dt,dt)
    v = -x0 * np.sin(t) + v0 * np.cos(t)
    x = x0 * np.cos(t) + v0 * np.sin(t)
    K = 1/2 *m*v**2
    U = 1/2 *k*x**2
    return x, v, K, U, t

def euler_cromer(k,m,x0,v0,dt,tfinal):
    n = np.ceil(tfinal/dt)\
    # Set up arrays
    t = np.zeros(n)
    v = np.zeros(n)
    x = np.zeros(n)
    K = np.zeros(n)
    U = np.zeros(n)
    # Define Initial Conditions
    x[0] = x0
    v[0] = v0
    K[0] = 1/2 *m*v0**2
    U[0] = 1/2 *k*x0**2

    # Integrate using the Euler-Cromer Method
    for i in range(n-1):
        a = -x[i]
        v[i+1] = v[i] + dt*a
        x[i+1] = x[i] + dt*v[i+1]
        K[i+1] = 1/2 *m*v[i+1]**2
        U[i+1] = 1/2 *k*x[i+1]**2
        t[i+1] = t[i] + dt
    return x, v, K, U, t

def velocity_verlet(k,m,x0,v0,dt,tfinal):
    n = np.ceil(tfinal/dt)
    # Set up arrays
    t = np.zeros(n)
    v = np.zeros(n)
    x = np.zeros(n)
    K = np.zeros(n)
    U = np.zeros(n)
    # Define Initial Conditions
    x[0] = x0
    v[0] = v0
    K[0] = 1/2 *m*v0**2
    U[0] = 1/2 *k*x0**2

    # Integrate using the Velocity-Verlet Method
    for i in range(n-1):
        a = -x[i]
        x[i+1] = x[i] + dt*v[i] + dt**2 /2*a
        a1 = -x[i+1]
        v[i+1] = v[i] + dt/2*(a+a1)
        K[i+1] = 1/2 *m*v[i+1]**2
        U[i+1] = 1/2 *k*x[i+1]**2
        t[i+1] = t[i] + dt
    return x, v, K, U, t

# Define Constants
dt = 0.01
tfinal = 50
x0 = 1
v0 = 0
m = 1
k = 1

# Call the Integration Function
ax, av, aK, aU, at = analytical(k,m,x0,v0,dt,tfinal)
ecx, ecv, ecK, ecU, ect = euler_cromer(k,m,x0,v0,dt,tfinal)
vvx, vvv, vvK, vvU,vvt = velocity_verlet(k,m,x0,v0,dt,tfinal)

# Plots
fig, axes = plt.subplots(1,3, figsize = (15,5))
fig.suptitle("System of an Undamped Spring", fontsize=14, y = 1.05)

axes[0].plot(at, ax, label = "Analytical Method")
axes[0].plot(ect, ecx, label = "Euler-Cromer Method")
axes[0].plot(vvt, vvx, label = "Velocity-Verlet Method")
axes[0].set_title("Position as a Function of Dimensionless Time")
axes[0].set_xlabel("Dimensionless Time")
axes[0].set_ylabel("Position")

axes[1].plot(at, av, label = "Analytical Method")
axes[1].plot(ect, ecv, label = "Euler-Cromer Method")
axes[1].plot(vvt, vvv, label = "Velocity-Verlet Method")
axes[1].set_title("Velocity as a Function of Dimensionless Time")
axes[1].set_xlabel("Dimensionless Time")
axes[1].set_ylabel("Velocity")

axes[2].plot(at, aU+aK, label = "Analytical Method")
axes[2].plot(ect, ecU+ecK, label = "Euler-Cromer Method")
axes[2].plot(vvt, vvU+vvK, label = "Velocity-Verlet Method")
axes[2].set_title("Energy as a Function of Dimensionless Time")
axes[2].set_xlabel("Dimensionless Time")
axes[2].set_ylabel("Total Energy")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()


def damp(gamma,m,x0,v0,DeltaT,tfinal):
    n = np.ceil(tfinal/DeltaT)
    # Set up arrays
    t = np.zeros(n)
    v = np.zeros(n)
    r = np.zeros(n)
    # Define Initial Conditions
    r[0] = x0
    v[0] = v0
    # Integrate over using the Velocity Verlet Method
    for i in range(n-1):
        a = -r[i] - 2*gamma*v[i]
        r[i+1] = r[i] + DeltaT*v[i] + DeltaT**2 /2 *a
        a1 = -r[i+1] - 2*gamma*v[i]
        v[i+1] = v[i] + DeltaT/2*(a+a1)
        t[i+1] = t[i] + DeltaT
        
    return t,r,v

# Define Constants
under_gamma = 0.1 
crit_gamma = 1
over_gamma = 2
m = 1
x0 = 1
v0 = 0
dt  = 0.1
tfinal = 50

# Call the Integration Function
t, under_r, under_v = damp(under_gamma,m,x0,v0,dt,tfinal)
t, crit_r, crit_v = damp(crit_gamma,m,x0,v0,dt,tfinal)
t, over_r, over_v = damp(over_gamma,m,x0,v0,dt,tfinal)


# Plots
fig, axes = plt.subplots(1,2, figsize = (15,5))
fig.suptitle("System of a Damped Spring", fontsize=14, y = 1.05)

axes[0].plot(t, under_r, label = "Under Damping")
axes[0].plot(t, crit_r, label = "Critical Damping")
axes[0].plot(t, over_r, label = "Over Damping")
axes[0].set_title("Position as a Function of Dimensionless Time")
axes[0].set_xlabel("Dimensionless Time")
axes[0].set_ylabel("Position")

axes[1].plot(t, under_v, label = "Under Damping")
axes[1].plot(t, crit_v, label = "Critical Damping")
axes[1].plot(t, over_v, label = "Over Damping")
axes[1].set_title("Velocity as a Function of Dimensionless Time")
axes[1].set_xlabel("Dimensionless Time")
axes[1].set_ylabel("Velocity")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()


def forced(gamma,r0,v0,F0,omega,d,DeltaT,tfinal):
    n = np.ceil(tfinal/DeltaT)
    # Set up arrays
    t = np.zeros(n)
    v = np.zeros(n)
    r = np.zeros(n)
    # Define Initial Conditions
    r[0] = r0
    v[0] = v0

    # Integrate using the 4th-Order RK Method
    for i in range(n-1):
        t[i+1] = t[i] + DeltaT
        
        Force = (-r[i] - 2*gamma*v[i] - F0*np.cos(omega*t[i]-d))*m
        k1x = DeltaT*v[i]
        k1v = DeltaT*Force
        
        vv = v[i]+k1v*0.5*DeltaT
        rr = r[i]+k1x*0.5*DeltaT
        Force = (-rr - 2*gamma*vv - F0*np.cos(omega*(t[i]+DeltaT*0.5)-d))*m
        k2x = DeltaT*vv
        k2v = DeltaT*Force
        
        vv = v[i]+k2v*0.5*DeltaT
        rr = r[i]+k2x*0.5*DeltaT
        Force = (-rr - 2*gamma*vv - F0*np.cos(omega*(t[i]+DeltaT*0.5)-d))*m
        k3x = DeltaT*vv
        k3v = DeltaT*Force

        vv = v[i]+k3v*DeltaT
        rr = r[i]+k3x*DeltaT
        Force = (-rr - 2*gamma*vv - F0*np.cos(omega*(t[i]+DeltaT*0.5)-d))*m
        k4x = DeltaT*vv
        k4v = DeltaT*Force
        
        r[i+1] = r[i]+(k1x+2*k2x+2*k3x+k4x)/6.
        v[i+1] = v[i]+(k1v+2*k2v+2*k3v+k4v)/6.
        
    return t,r,v

# Define Constants
gamma = 0.1
r0 = 1
v0 = 0
F0 = 3
omega = 3
d = 0
dt = 0.1
tfinal = 100

# Call the Integration Function
t, r, v = forced(gamma,r0,v0,F0,omega,d,dt,tfinal)

# Plots
fig, axes = plt.subplots(1,2, figsize = (15,5))
fig.suptitle("System of a Damped Spring with a Driving Force", fontsize=14, y = 1.05)

axes[0].plot(t, r)
axes[0].set_title("Position as a Function of Dimensionless Time")
axes[0].set_xlabel("Dimensionless Time")
axes[0].set_ylabel("Position")

axes[1].plot(t, v)
axes[1].set_title("Velocity as a Function of Dimensionless Time")
axes[1].set_xlabel("Dimensionless Time")
axes[1].set_ylabel("Velocity")

plt.tight_layout()
