import numpy as np
import matplotlib # do I need this?
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Ising_Model(): # do I need a class?

    def __init__(self, J: float=1.0):
        self.J = J

    def energy(self, spin_config):
        product_x = np.roll(spin_config, 1, axis=1) * spin_config
        product_y = np.roll(spin_config, 1, axis=0) * spin_config
        energy = -self.J * (np.sum(product_x) + np.sum(product_y))
        return energy


class MCMC():

    def __init__(self, L: int=3):
        self.L = L
        self.thermalization_time = 10 * L * L
        self.corr_time = L * L

    def flip(self, spin_config):
        index = np.random.randint(spin_config.shape[0], size=2)
        spin_config[index[0], index[1]] = -spin_config[index[0], index[1]] # not so pretty yet
        return spin_config

    def sample(self, model, beta: float=1.0, N_mc: int=5):
        spin_config = np.random.choice([-1, 1], size=(self.L, self.L))
        config_energy = model.energy(spin_config)

        config_list = []
        energy_list = []
        i = 0
        while len(energy_list) < N_mc:
            new_spin_config = self.flip(np.copy(spin_config)) # not very nice
            new_config_energy = model.energy(new_spin_config)

            prob_ratio = np.exp(-beta * (new_config_energy - config_energy))

            if np.random.random() < prob_ratio: # the case new_config_energy <= config_energy is automatically included
                spin_config = new_spin_config
                config_energy = new_config_energy

            if i >= self.thermalization_time:
                if i % self.corr_time == 0:
                    config_list.append(spin_config)
                    energy_list.append(config_energy)
            i += 1

        return energy_list, config_list

def magnetization_density(spin_configs):
    magnetization_density = [np.sum(config) / config.size for config in spin_configs]
    magnetization_av = np.mean(magnetization_density)
    magnetization_var = np.var(magnetization_density)
    return magnetization_av, magnetization_var

def energy_density(energies, L = 5):
    energy_av = np.mean(energies) / (L**2)
    energy_var = np.var(energies) / (L**4)
    return energy_av, energy_var


temperatures = np.linspace(1.0, 4.0, 7)
L = 40
J = 1.0
N_mc = 1e3

ising = Ising_Model(J=J)
mcmc = MCMC(L=L)

magnetization_avs = []
magnetization_vars = []
energy_avs = []
energy_vars = []
for T in temperatures:
    energies, configs = mcmc.sample(ising, beta=1/T, N_mc=N_mc)

    np.save('spins_T%s.npy' % T, configs) # np.load(string + ".npy")
    np.save('energies_T%s.npy' % T, energies) # np.load(string + ".npy")

    magnetization_density = [np.sum(config) / config.size for config in configs]
    magnetization_avs.append(np.mean(magnetization_density))
    magnetization_vars.append(np.var(magnetization_density))

    energy_avs.append(np.mean(energies) / (L**2))
    energy_vars.append(np.var(energies) / (L**4))


plt.rc('font', family='serif')#, size=14)
plt.rc('text', usetex=True)
#plt.rc('xtick', labelsize=14)
#plt.rc('ytick', labelsize=14)
#plt.rc('axes', labelsize=14)

#ax.xaxis.set_major_locator(MultipleLocator(0.5))
#ax.xaxis.set_minor_locator(MultipleLocator(0.1))

#ax.yaxis.set_major_locator(MultipleLocator(0.01))
#ax.yaxis.set_minor_locator(MultipleLocator(0.002))

plt.figure()
plt.title('Average magnetization')
plt.xlabel(r'$T$')
plt.ylabel(r'|$\langle M \rangle|$')
plt.plot(temperatures, np.abs(magnetization_avs))
plt.tight_layout()
plt.savefig('magnetization_av.pdf')
plt.close()

plt.figure()
plt.title('Magnetization variance')
plt.xlabel(r'$T$')
plt.ylabel(r'$\langle M^2 \rangle - \langle M \rangle^2$')
plt.plot(temperatures, magnetization_vars)
plt.tight_layout()
plt.savefig('magnetization_var.pdf')
plt.close()

plt.figure()
plt.title('Average energy')
plt.xlabel(r'$T$')
plt.ylabel(r'$\langle H \rangle$')
plt.plot(temperatures, energy_avs)
plt.tight_layout()
plt.savefig('energy_av.pdf')
plt.close()

plt.figure()
plt.title('Energy variance')
plt.xlabel(r'$T$')
plt.ylabel(r'$\langle H^2 \rangle - \langle H \rangle^2$')
plt.plot(temperatures, energy_vars)
plt.tight_layout()
plt.savefig('energy_var.pdf')
plt.close()
