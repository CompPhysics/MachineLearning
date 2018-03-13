# start import modules
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import emcee
# end import modules

savefig=True

# start generate data
np.random.seed(1)      # for repeatability
F_true = 1000          # true flux, say number of photons measured in 1 second
N = 50                 # number of measurements
F = stats.poisson(F_true).rvs(N)
                       # N measurements of the flux
e = np.sqrt(F)         # errors on Poisson counts estimated via square root
# end generate data

# start visualize data
fig, ax = plt.subplots()
ax.errorbar(F, np.arange(N), xerr=e, fmt='ok', ecolor='gray', alpha=0.5)
ax.vlines([F_true], 0, N, linewidth=5, alpha=0.2)
ax.set_xlabel("Flux");ax.set_ylabel("measurement number");
# end visualize data

if savefig:
    fig.savefig('../fig/singlephotoncount_fig_1.png')

# start frequentist
w=1./e**2
print("""
F_true = {0}
F_est = {1:.0f} +/- {2:.0f} (based on {3} measurements) """\
          .format(F_true, (w * F).sum() / w.sum(), w.sum() ** -0.5, N))
# end frequentist

# start bayesian setup
def log_prior(alpha):
    return 0 # flat prior

def log_likelihood(alpha, F, e):
    return -0.5 * np.sum(np.log(2 * np.pi * e ** 2) \
                             + (F - alpha[0]) ** 2 / e ** 2)
                             
def log_posterior(alpha, F, e):
    return log_prior(alpha) + log_likelihood(alpha, F, e)
# end bayesian setup

# start bayesian mcmc
ndim = 1      # number of parameters in the model
nwalkers = 50 # number of MCMC walkers
nburn = 1000  # "burn-in" period to let chains stabilize
nsteps = 2000 # number of MCMC steps to take
# we'll start at random locations between 0 and 2000
starting_guesses = 2000 * np.random.rand(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[F,e])
sampler.run_mcmc(starting_guesses, nsteps)
# Shape of sampler.chain  = (nwalkers, nsteps, ndim)
# Flatten the sampler chain and discard burn-in points:
samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
# end bayesian mcmc

# start visualize bayesian
fig, ax = plt.subplots()
ax.hist(samples, bins=50, histtype="stepfilled", alpha=0.3, normed=True)
ax.set_xlabel(r'$F_\mathrm{est}$')
ax.set_ylabel(r'$p(F_\mathrm{est}|D,I)$')
# end visualize bayesian

if savefig:
    fig.savefig('../fig/singlephotoncount_fig_2.png')

# plot a best-fit Gaussian
F_est = np.linspace(975, 1025)
pdf = stats.norm(np.mean(samples), np.std(samples)).pdf(F_est)
ax.plot(F_est, pdf, '-k')

# start bayesian CI
sampper=np.percentile(samples, [2.5, 16.5, 50, 83.5, 97.5],axis=0).flatten()
print("""
F_true = {0}
Based on {1} measurements the posterior point estimates are:
...F_est = {2:.0f} +/- {3:.0f}
or using credible intervals:
...F_est = {4:.0f}          (posterior median) 
...F_est in [{5:.0f}, {6:.0f}] (67% credible interval) 
...F_est in [{7:.0f}, {8:.0f}] (95% credible interval) """\
          .format(F_true, N, np.mean(samples), np.std(samples), \
                      sampper[2], sampper[1], sampper[3], sampper[0], sampper[4]))
# end bayesian CI

if not savefig:
    plt.show()

