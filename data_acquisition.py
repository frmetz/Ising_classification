import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)


def energy(spin_config, J=1.0):
    """Calculates energy of Ising model given spin configuration"""
    product_x = np.roll(spin_config, 1, axis=1) * spin_config
    product_y = np.roll(spin_config, 1, axis=0) * spin_config
    ising_energy = -J * (np.sum(product_x) + np.sum(product_y))
    return ising_energy


def flip(spin_config):
    """Flips a single spin at random position"""
    index = tuple(np.random.randint(spin_config.shape[0], size=2))
    spin_config[index] = -spin_config[index]
    return spin_config


def mcmc_sample(beta, N_mc=1, L=40):
    """MCMC for generating spin configurations at a given temperature"""
    thermalization_time = L ** 4
    corr_time = 10 * L ** 2

    spin_config = np.random.choice([-1, 1], size=(L, L))
    config_energy = energy(spin_config)

    config_list = []
    energy_list = []
    i = 0
    while len(energy_list) < N_mc:
        new_spin_config = flip(np.copy(spin_config))
        new_config_energy = energy(new_spin_config)

        prob_ratio = np.exp(-beta * (new_config_energy - config_energy))

        # the case new_config_energy <= config_energy is automatically included
        if np.random.random() < prob_ratio:
            spin_config = new_spin_config
            config_energy = new_config_energy

        if i >= thermalization_time:
            if i % corr_time == 0:
                config_list.append(spin_config)
                energy_list.append(config_energy)
        i += 1

    return energy_list, config_list


def save_data(temperatures):
    """Generates and saves spin configurations at different temperatures"""
    L = 40
    N_mc = 5e3  # 5000 except for critical region

    for T in temperatures:
        print(T)
        energies, configs = mcmc_sample(beta=1 / T, N_mc=N_mc, L=L)

        np.save("data/spins_T%s.npy" % T, configs)
        np.save("data/energies_T%s.npy" % T, energies)


def plot_data(temperatures):

    magnetization_avs = []
    magnetization_vars = []
    energy_avs = []
    energy_vars = []
    for T in temperatures:
        configs = np.load("data/spins_T%s.npy" % T)
        energies = np.load("data/energies_T%s.npy" % T)

        magnetization_density = np.abs(
            np.array([np.sum(config) / config.size for config in configs])
        )
        magnetization_avs.append(np.mean(magnetization_density))
        magnetization_vars.append(np.var(magnetization_density))
        L = configs.shape[1]
        energy_avs.append(np.mean(energies) / (L ** 2))
        energy_vars.append(np.var(energies) / (L ** 4))

    plt.rc("font", family="serif", size=14)
    plt.rc("text", usetex=True)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', labelsize=14)

    plt.figure()
    plt.title("(a) Average magnetization")
    plt.xlabel(r"$T$")
    plt.ylabel(r"$|\langle M \rangle|$")
    plt.plot(temperatures, magnetization_avs)
    plt.tight_layout()
    plt.savefig("plots/magnetization_av.pdf")
    plt.close()

    plt.figure()
    plt.title("(c) Magnetic susceptibility")
    plt.xlabel(r"$T$")
    plt.ylabel(r"$\langle M^2 \rangle - \langle M \rangle^2$")
    plt.plot(temperatures, magnetization_vars)
    plt.tight_layout()
    plt.savefig("plots/magnetization_var.pdf")
    plt.close()

    plt.figure()
    plt.title("(b) Average energy")
    plt.xlabel(r"$T$")
    plt.ylabel(r"$\langle H \rangle$")
    plt.plot(temperatures, energy_avs)
    plt.tight_layout()
    plt.savefig("plots/energy_av.pdf")
    plt.close()

    plt.figure()
    plt.title("(d) Energy variance")
    plt.xlabel(r"$T$")
    plt.ylabel(r"$\langle H^2 \rangle - \langle H \rangle^2$")
    plt.plot(temperatures, energy_vars)
    plt.tight_layout()
    plt.savefig("plots/energy_var.pdf")
    plt.close()


#temperatures = [1.0, 1.5, 3.0, 3.5, 4.0]
#temperatures = np.linspace(1.0, 4.0, 7)
# save_data(temperatures)
# plot_data(temperatures)
