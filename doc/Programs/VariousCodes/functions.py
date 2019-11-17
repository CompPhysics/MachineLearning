import sys
import numpy as np
from matplotlib import cm
"""
A file for all common functions used in project 1
"""


def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

def MSE(y, y_tilde):
	"""
	Function for computing mean squared error.
	Input is y: analytical solution, y_tilde: computed solution.
	"""
	return np.sum((y-y_tilde)**2)/y.size

def R2_Score(y, y_tilde):
	"""
	Function for computing the R2 score.
	Input is y: analytical solution, y_tilde: computed solution.
	"""

	return 1 - np.sum((y[:-2]-y_tilde[:-2])**2)/np.sum((y[:-2]-np.average(y))**2)

def create_X(x, y, n = 5):
	"""
	Function for creating a X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
	Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polinomial you want to fit.
	"""
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X


def plot_surface(x, y, z, title = "", show = False, cmap=cm.coolwarm, figsize = None):
	"""
	Function to plot surfaces of z, given an x and y.
	Input: x, y, z (NxN'Modeler' matrices), and a title (string)
	"""

	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.pyplot as plt
	from matplotlib.ticker import LinearLocator, FormatStrFormatter

	if figsize:
		fig = plt.figure(figsize = figsize)
	else:
		fig = plt.figure()
	
	ax = fig.gca(projection='3d')

	# Plot the surface.of the best fit

	surf = ax.plot_surface(x, y, z, cmap=cmap,
				   linewidth=0, antialiased=False)


	# Customize the z axis automatically
	z_min = np.min(z)
	z_min = z_min*1.01 if z_min < 0 else z_min*.99
	z_max = np.max(z)
	z_max = z_max*1.01 if z_max > 0 else z_max*.99

	ax.set_zlim(z_min, z_max)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	ax.view_init(azim=20,elev=45)
	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)
	ax.set_title(title)

	if show:
		plt.show()

	return fig, ax ,surf

def train_test_data(x_,y_,z_,i):
	"""
	Takes in x,y and z arrays, and a array with random indesies iself.
	returns learning arrays for x, y and z with (N-len(i)) dimetions
	and test data with length (len(i))
	"""
	x_learn=np.delete(x_,i)
	y_learn=np.delete(y_,i)
	z_learn=np.delete(z_,i)
	x_test=np.take(x_,i)
	y_test=np.take(y_,i)
	z_test=np.take(z_,i)

	return x_learn,y_learn,z_learn,x_test,y_test,z_test


def K_fold(x,y,z,k,alpha,model,m=5, ret_std = False):
	"""Function to who calculate the average MSE and R2 using k-fold.
	Takes in x,y and z varibles for a dataset, k number of folds, alpha and which method beta shall use. (OLS,Ridge or Lasso)
	Returns average MSE and average R2"""
	print(m)
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)
		z = np.ravel(z)
	n=len(x)
	n_k=int(n/k)
	if n_k*k!=n:
		print("k needs to be a multiple of ", n,k)
	i=np.arange(n)
	np.random.shuffle(i)

	MSE_=0
	R2_=0
	Variance_=0
	Bias_=0
	betas = np.zeros((k,int((m+1)*(m+2)/2)))
	for t in range(k):
		x_,y_,z_,x_test,y_test,z_test=train_test_data(x,y,z,i[t*n_k:(t+1)*n_k])
		X= create_X(x_,y_,n=m)
		X_test= create_X(x_test,y_test,n=m)


		model.fit(X,z_)
		betas[t] = model.beta
		z_predict=model.predict(X_test)

		MSE_+=MSE(z_test,z_predict)
		R2_+=R2_Score(z_test,z_predict)
		Bias_+=bias(z_test,z_predict)
		Variance_+=variance(z_predict)

	return (MSE_/k, R2_/k, Bias_/k, Variance_/k, np.std(betas, axis = 0), np.mean(betas, axis = 0))


def variance(y_tilde):
	"""
	Calculates the variance of the predicted values y_tilde.
	"""
	return np.sum((y_tilde - np.mean(y_tilde))**2)/np.size(y_tilde)

def bias(y, y_tilde):
	"""
	Calculates the bias of the predicted values y_tilde compared to
	the actual data y.
	"""
	return np.sum((y - np.mean(y_tilde))**2)/np.size(y_tilde)


def update_progress(job_title, progress):
	"""
	Shows the progress of an for-loop.
	"""
	length = 20 # modify this to change the length
	block = int(round(length*progress))
	msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
	if progress >= 1: msg += " DONE\r\n"
	sys.stdout.write(msg)
	sys.stdout.flush()


def savefigure(name, figure = "gcf"):
	"""
	Function for saving figures as a .tex-file for easier integration with latex.
	"""
	try:
		from matplotlib2tikz import save as tikz_save
		tikz_save(name.replace(" ", "_") + ".tex", figure = figure, figureheight='\\figureheight', figurewidth='\\figurewidth')
	except ImportError:
		print("Please install matplotlib2tikz to save figure as a .tex-file.")
		import matplotlib.pyplot as plt
		if figure == "gcf":
			plt.savefig(name+".pdf")
		else:
			fig.savefig(name+".pdf")
