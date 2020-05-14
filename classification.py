import time

import jax.numpy as jnp
from jax.config import config
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax
from jax.experimental.stax import (BatchNorm, Conv, Dense, Flatten,
                                   Relu, LogSoftmax, Softmax, Sigmoid)

import matplotlib.pyplot as plt
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
    losses = loss(params, batch)
    return opt_update(i, grad(loss)(params, batch), opt_state), losses

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

train_acc_list = []
test_acc_list = []
loss_list = []

train_acc_list.append(accuracy(init_params, (train_images, train_labels)))
test_acc_list.append(accuracy(init_params, (test_images, test_labels)))

opt_state = opt_init(init_params)
itercount = 0

print("\nStarting training...")
for epoch in range(num_epochs):
    start_time = time.time()
    for _ in range(num_batches):
        opt_state, losses = update(itercount, opt_state, next(batches))
        loss_list.append(losses)
        itercount += 1
    epoch_time = time.time() - start_time

    params = get_params(opt_state)
    train_acc = accuracy(params, (train_images, train_labels))
    test_acc = accuracy(params, (test_images, test_labels))
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))


plt.rc('font', family='serif')#, size=14)
plt.rc('text', usetex=True)
#plt.rc('xtick', labelsize=14)
#plt.rc('ytick', labelsize=14)
#plt.rc('axes', labelsize=14)
#plt.rcParams['axes.labelsize']  = 10
#plt.rcParams['legend.fontsize'] = 10
#plt.rcParams['xtick.labelsize'] = 8
#plt.rcParams['ytick.labelsize'] = 8

x = jnp.linspace(0, num_epochs, num_epochs+1)
plt.figure()
plt.title('Training curve')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.plot(x, train_acc_list, label="train accuracy")
plt.plot(x, test_acc_list, label="test accuracy")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plots/training_accuracy.pdf')
plt.close()

plt.figure()
plt.title('Training curve')
plt.xlabel('update steps')
plt.ylabel('Loss')
plt.plot(loss_list)
plt.tight_layout()
plt.savefig('plots/training_loss.pdf')
plt.close()
