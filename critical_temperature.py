import matplotlib.pyplot as plt
import numpy as np

import jax.numpy as jnp

from data_acquisition import *
from classification import Classifier, plot_training


plt.rc("font", family="serif", size=14)
plt.rc("text", usetex=True)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=14)

CNN = True
L = 40
T_critical = 2 / np.log(1+np.sqrt(2))

ising_classifier = Classifier(CNN=CNN)
#plot_training(*ising_classifier.train()) # new training
ising_classifier.params = jnp.load('data/params.npy', allow_pickle=True) # use already trained model


############################################
# Plot classification of train+test data
############################################
temperatures = np.linspace(1.0, 4.0, 7)
mean = []
std = []

plt.figure()
plt.title('Classification results')
plt.xlabel(r'$T$')
plt.ylabel('Prediction')
for T in temperatures:
    configs = np.load('data/spins_T%s.npy' % T).reshape((-1, L*L)).astype(np.float64)
    magnetization_density = np.abs(np.array([np.sum(config) / config.size for config in configs]))
    color = np.where(magnetization_density < 0.5, 'tab:blue', 'tab:green')
    if CNN:
        configs = configs.reshape((-1, L, L, 1))
    prediction = ising_classifier.classify(configs).reshape(-1)
    mean.append(np.mean(prediction))
    std.append(np.std(prediction))
    plt.scatter([T]*prediction.size, prediction, color=color)
    plt.errorbar([T], np.mean(prediction), yerr=np.std(prediction), fmt='o', color='tab:orange')
plt.tight_layout()
plt.savefig('plots/classification.pdf')
plt.close()

############################################
# Find critical temperature (fine resolution)
############################################
temperatures = np.linspace(2.2, 2.35, 15) # finer resolution in critical region
N_mc = 100

mean2 = []
std2 = []
for i, T in enumerate(temperatures):
    # generate data at given T
    #_, configs = mcmc_sample(beta=1 / T, N_mc=N_mc, L=L)
    #configs = np.array(configs).reshape((-1, L*L)).astype(float)
    #np.save("data/spins_val_T%s.npy" % T, configs)

    # use already saved data
    configs = np.load('data/spins_val_T%s.npy' % i)

    magnetization_density = np.abs(np.array([np.sum(config) / config.size for config in configs]))
    if CNN:
        configs = configs.reshape((-1, L, L, 1))
    prediction = ising_classifier.classify(configs).reshape(-1)
    mean2.append(np.mean(prediction))
    std2.append(np.std(prediction))

plt.figure()
plt.title('Critical Region')
plt.xlabel(r'$T$')
plt.ylabel('Prediction')
plt.errorbar(temperatures, mean2, yerr=std2, fmt='o', color='tab:blue', linestyle='-')
plt.axvline(x=T_critical, color='tab:orange', label="exact value")
plt.axhline(y=0.5, color='black', linestyle='--')
#plt.axvline(x=2.294, color='tab:green', alpha=0.5, linewidth=20)
plt.axvline(x=2.294, color='tab:green', label="prediction")
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('plots/critical_temperature.pdf')
plt.close()

############################################
# Find critical temperature (full temperature range)
############################################
temperatures1 = np.array([1.0, 1.5, 2.0])
temperatures2 = np.array([2.5, 3.0, 3.5, 4.0])
temperatures = np.concatenate((temperatures1, temperatures[[0, -1]], temperatures2))
mean = np.array(mean)
std = np.array(std)
mean2 = np.array(mean2)
std2 = np.array(std2)
means = np.concatenate((mean[:3], mean2[[0, -1]], mean[3:]))
stds = np.concatenate((std[:3], std2[[0, -1]], std[3:]))
plt.figure()
plt.title('Critical Region')
plt.xlabel(r'$T$')
plt.ylabel('Prediction')
plt.errorbar(temperatures, means, yerr=stds, fmt='o', color='tab:blue', linestyle='-')
plt.axvline(x=T_critical, color='tab:orange', label="exact value")
plt.axhline(y=0.5, color='black', linestyle='--')
plt.axvline(x=2.28, color='tab:green', label="prediction")
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('plots/critical_temperature2.pdf')
plt.close()
