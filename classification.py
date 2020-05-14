import time

import jax.numpy as jnp
from jax.config import config
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax
from jax.experimental.stax import (BatchNorm, Conv, Dense, Flatten,
                                   Relu, LogSoftmax, Softmax, Sigmoid)

from data_preprocessing import data_preprocessing


def loss(params, batch): # done
    inputs, targets = batch
    preds = predict(params, inputs)
    return jnp.mean(jnp.nan_to_num(-targets*jnp.log(preds)-(1-targets)*jnp.log(1-preds))) # cross-entropy loss

def accuracy(params, batch): # done
    inputs, target_class = batch
    predicted_class = jnp.where(predict(params, inputs) < 0.5, 0, 1)
    return jnp.mean(predicted_class == target_class)

@jit
def update(i, opt_state, batch): # done
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)

def data_stream():
    # num_train, num_batches, batch_size, train_images, train_labels
    rng = random.PRNGKey(0)
    while True:
        perm = random.permutation(rng, num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
            yield train_images[batch_idx], train_labels[batch_idx]


init_random_params2, predict2 = stax.serial(Conv(out_chan=10, filter_shape=(2, 2), strides=(1, 1), padding="VALID"),
                                 BatchNorm(), Relu,
                                 #Conv(10, (3, 3), (1, 1), padding="SAME"), Relu,
                                 Flatten,
                                 #Dense(64),
                                 #Relu,
                                 Dense(1),
                                 #LogSoftmax)
                                 Softmax)


init_random_params, predict = stax.serial(Conv(out_chan=64, filter_shape=(2, 2), strides=(1, 1), padding="VALID"),
                                    Relu,
                                    Flatten,
                                    Dense(64) , Relu, Dense(1),
                                    Sigmoid)




rng = random.PRNGKey(0)

step_size = 0.001
num_epochs = 10
batch_size = 128
momentum_mass = 0.9
L = 40

#train_images, train_labels, test_images, test_labels = data_preprocessing()

train_images = jnp.load('data/x_test.npy')
train_labels = jnp.load('data/y_test.npy')
test_images = jnp.load('data/x_train.npy')
test_labels = jnp.load('data/y_train.npy')

num_train = train_images.shape[0]
num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = num_complete_batches + bool(leftover)



batches = data_stream()

opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)
#opt_init, opt_update, get_params = optimizers.adam(0.0001)

_, init_params = init_random_params(rng, (-1, L, L, 1))


print(accuracy(init_params, (train_images, train_labels)))
print(accuracy(init_params, (test_images, test_labels)))

opt_state = opt_init(init_params)
itercount = 0

print("\nStarting training...")
for epoch in range(num_epochs):
    start_time = time.time()
    for _ in range(num_batches):
        opt_state = update(itercount, opt_state, next(batches))
        itercount += 1
    epoch_time = time.time() - start_time

    params = get_params(opt_state)
    train_acc = accuracy(params, (train_images, train_labels))
    test_acc = accuracy(params, (test_images, test_labels))
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))
