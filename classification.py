import time
from functools import partial


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
from architectures import PeriodicConv

class Classifier():

    def __init__(self, CNN="true", step_size=0.001, seed=0):

        self.rng = random.PRNGKey(seed)

        self.CNN = CNN
        L = 40
        if CNN:
            self.input_shape = (-1, L, L, 1)
            self.init_random_params, self.predict = stax.serial(
                                                                #Conv(out_chan=64, filter_shape=(2, 2), strides=(1, 1), padding="VALID"),
                                                                PeriodicConv(out_chan=64, filter_shape=(2, 2), strides=(1, 1), padding='VALID'),
                                                                Relu,
                                                                Flatten,
                                                                Dense(64),
                                                                Relu,
                                                                Dense(1),
                                                                Sigmoid
                                                                )
        else:
            self.input_shape = (-1, L*L)
            self.init_random_params, self.predict = stax.serial(
                                                                Dense(100),
                                                                Relu,
                                                                Dense(100),
                                                                Relu,
                                                                Dense(1),
                                                                Sigmoid
                                                                )



        momentum_mass = 0.9
        self.opt_init, self.opt_update, self.get_params = optimizers.momentum(step_size, mass=momentum_mass)
        #self.opt_init, self.opt_update, self.get_params = optimizers.adam(0.0001)

        _, self.init_params = self.init_random_params(self.rng, self.input_shape)
        self.opt_state = self.opt_init(self.init_params)
        self.params = self.init_params


    def loss(self, params, batch): # done
        inputs, targets = batch
        preds = self.predict(params, inputs)
        return jnp.mean(jnp.nan_to_num(-targets*jnp.log(preds)-(1-targets)*jnp.log(1-preds))) # cross-entropy loss

    def accuracy(self, params, batch): # done
        inputs, target_class = batch
        predicted_class = jnp.where(self.predict(params, inputs) < 0.5, 0, 1)
        return jnp.mean(predicted_class == target_class)

    def classify(self, inputs):
        return self.predict(self.params, inputs)

    @partial(jit, static_argnums=(0,))
    def update(self, i, opt_state, batch): # done
        params = self.get_params(opt_state)
        losses = self.loss(params, batch)
        return self.opt_update(i, grad(self.loss)(params, batch), opt_state), losses

    def data_stream(self, train_images, train_labels, num_train, num_batches, batch_size): # done
        while True:
            perm = random.permutation(self.rng, num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]


    def train(self, num_epochs = 10, batch_size = 128):

        #train_images, train_labels, test_images, test_labels = data_preprocessing()

        train_images = jnp.load('data/x_test.npy')
        train_labels = jnp.load('data/y_test.npy')
        test_images = jnp.load('data/x_train.npy')
        test_labels = jnp.load('data/y_train.npy')

        if not self.CNN:
            train_images = jnp.reshape(train_images, self.input_shape)
            test_images = jnp.reshape(test_images, self.input_shape)

        num_train = train_images.shape[0]
        num_complete_batches, leftover = divmod(num_train, batch_size)
        num_batches = num_complete_batches + bool(leftover)
        batches = self.data_stream(train_images, train_labels, num_train, num_batches, batch_size)


        train_acc_list = []
        test_acc_list = []
        loss_list = []

        train_acc_list.append(self.accuracy(self.init_params, (train_images, train_labels)))
        test_acc_list.append(self.accuracy(self.init_params, (test_images, test_labels)))

        opt_state = self.opt_init(self.init_params)
        itercount = 0

        print("\nStarting training...")
        for epoch in range(num_epochs):
            start_time = time.time()
            for _ in range(num_batches):
                opt_state, losses = self.update(itercount, opt_state, next(batches))
                loss_list.append(losses)
                itercount += 1
            epoch_time = time.time() - start_time

            params = self.get_params(opt_state)
            train_acc = self.accuracy(params, (train_images, train_labels))
            test_acc = self.accuracy(params, (test_images, test_labels))
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
            print("Training set accuracy {}".format(train_acc))
            print("Test set accuracy {}".format(test_acc))

        self.opt_state = opt_state
        self.params = params
        return train_acc_list, test_acc_list, loss_list

def plot_training(train_acc_list, test_acc_list, loss_list):

    plt.rc('font', family='serif')#, size=14)
    plt.rc('text', usetex=True)
    #plt.rc('xtick', labelsize=14)
    #plt.rc('ytick', labelsize=14)
    #plt.rc('axes', labelsize=14)
    #plt.rcParams['axes.labelsize']  = 10
    #plt.rcParams['legend.fontsize'] = 10
    #plt.rcParams['xtick.labelsize'] = 8
    #plt.rcParams['ytick.labelsize'] = 8

    x = jnp.linspace(0, len(train_acc_list)-1, len(train_acc_list))
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


#ising_classifier = Classifier()
#train_acc_list, test_acc_list, loss_list = ising_classifier.train()
#plot_training(train_acc_list, test_acc_list, loss_list)
