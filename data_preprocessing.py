import jax.numpy as jnp
from jax import random

def data_preprocessing():

    rng = random.PRNGKey(0)

    temperatures = jnp.linspace(1.0, 4.0, 7)
    temperatures1 = [1.0, 1.5, 3.0, 3.5, 4.0]
    temperatures2 = [2.0, 2.5]

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for T in temperatures:
        configs = jnp.load('data/spins_T%s.npy' % T)
        magnetization_density = jnp.abs(jnp.array([jnp.sum(config) / config.size for config in configs]))
        labels = jnp.where(magnetization_density < 0.5, 0, 1)
        if T in temperatures2:
            x_test.append(configs)
            y_test.append(labels)
        else:
            indices = random.permutation(rng, labels.size) # before seperating into test and training set, I shuffle the data to make sure I am not picking up some spurious correlations between successive configurations
            y_test.append(labels[indices[:int(0.2*labels.size)]])
            y_train.append(labels[indices[int(0.2*labels.size):]])
            x_test.append(configs[indices[:int(0.2*labels.size)]])
            x_train.append(configs[indices[int(0.2*labels.size):]])


    y_test_new = jnp.array(y_test[0])
    x_test_new = jnp.array(x_test[0])
    for i in range(len(y_test)-1): # not very pretty
        y_test_new = jnp.concatenate((y_test_new, y_test[i+1]))
        x_test_new = jnp.concatenate((x_test_new, x_test[i+1]))

    L = jnp.array(x_train).shape[2]
    x_test = jnp.array(x_test_new).reshape((-1, L, L, 1))
    #x_test = jnp.array(x_test_new).reshape((-1, L*L))
    y_test = jnp.array(y_test_new).reshape((-1, 1))
    x_train = jnp.array(x_train).reshape((-1, L, L, 1))
    #x_train = jnp.array(x_train).reshape((-1, L*L))
    y_train = jnp.array(y_train).reshape((-1, 1))

    #(4000, 40, 40)
    #(4000, 1)
    #(3000, 40, 40)
    #(3000, 1)

    jnp.save('data/x_test.npy', x_test.astype(float)) # np.load(string + ".npy")
    jnp.save('data/y_test.npy', y_test)
    jnp.save('data/x_train.npy', x_train.astype(float))
    jnp.save('data/y_train.npy', y_train)

    return x_train.astype(float), y_train, x_test.astype(float), y_test


data_preprocessing()
