import os
import sys
from imageio import imread
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import Data 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pathlib as pl

def make_FrankeFunction(n=1000, linspace=False, noise_std=0, random_state=42):
	x, y = None, None

	np.random.seed(random_state)
	if linspace:
		perfect_square = ( int(n) == int(np.sqrt(n))**2)
		assert perfect_square, f"{n = } is not a perfect square. Thus linspaced points cannot be made"

		x = np.linspace(0, 1, int(np.sqrt(n)))
		y = np.linspace(0, 1, int(np.sqrt(n)))

		X, Y = np.meshgrid(x, y)
		x = X.flatten()
		y = Y.flatten()
	else:
		x = np.random.uniform(low=0, high=1, size=n)
		y = np.random.uniform(low=0, high=1, size=n)
		

	z = FrankeFunction(x, y) + np.random.normal(loc=0, scale=noise_std, size=n)

	return Data(z, np.c_[x,y])

def plot_surf(D):
	sns.set_style("white")
	fig = plt.figure()
	ax = fig.add_subplot(projection="3d")

	X = D.X
	if D.X.shape[1] == 2:
		surf = ax.plot_trisurf(*X.T, D.y, cmap=cm.viridis, linewidth=0, antialiased=False)
	else:
		surf = ax.plot_trisurf(X[:,1], X[:,2], D.y, cmap=cm.viridis, linewidth=0, antialiased=False)
	
	ax.set_xlabel("x", fontsize=14)
	ax.set_ylabel("y", fontsize=14)
	cbar = fig.colorbar(surf, shrink=0.5, aspect=5)

	return fig, ax, surf, cbar


def plot_FrankeFunction(D, angle=(18, 45), filename=None):
	fig, ax, surf, cbar = plot_surf(D)

	ax.set_zlim(-0.10, 1.40)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
	ax.set_zlabel(r"$F (x,y)$", fontsize=14, rotation=90)
	ax.view_init(*angle)

	if filename:
		plt.savefig(filename, dpi=300)
	
	fig.tight_layout()
	plt.show()

def plot_Terrain(D, angle=(18,45), figsize=(10,7), filename=None):
	fig, ax, surf, cbar = plot_surf(D)

	fig.set_size_inches(*figsize)
	ax.set_zlabel(r"Terrain", fontsize=14, rotation=90)
	ax.view_init(*angle)

	if filename:
		plt.savefig(filename, dpi=300)
	
	fig.tight_layout()
	plt.show()

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4


def load_Terrain(filename="SRTM_data_Nica.tif", n=900, random_state=321):
	path = pl.Path(__file__).parent / filename
	start, stop = 1600, 1900

	assert n <= (stop-start)**2, f"Cannot load {n} points of terrain data, maximum available is {(stop-start)**2}."

	z = imread(path)[start:stop, start:stop]
	
	# drawing random samples from grid
	np.random.seed(random_state)
	x1 = np.arange(stop=stop-start) # NS-coordinates
	x2 = x1.copy() # EW-coordinates
	# Making array of every combination of (x1, x2)
	X = np.reshape(np.meshgrid(x1, x2), (2, (stop-start)**2)).T
	np.random.shuffle(X) # shuffling for randomness
	X = X[:n] # drawing n points

	y = np.zeros(shape=n, dtype=float)
	for i, (x1,x2) in enumerate(X):
		y[i] = z[x1,x2]

	return Data(y, X.astype(float))


if __name__ == "__main__":
	# D = make_FrankeFunction(n=625, uniform=False, noise_std=0.1)
	# plot_FrankeFunction(D)

	D = load_Terrain(n = 9000)
	plot_Terrain(D, angle=(22,-55))
