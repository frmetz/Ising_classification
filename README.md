# Classifying the phases of the 2D Ising model

Phase classification of the 2D Ising model similiar to the paper [Machine learning phases of matter](https://doi.org/10.1038/nphys4035) using the machine learning library [JAX](https://jax.readthedocs.io/en/latest/).

This repository contains the following scripts:

* **data_acquisition.py** - Generates spin configurations at different temperatures using MCMC and plots quantities like magnetization, susceptibility, etc.
* **data_preprocessing.py** - Divides data into train and test set and generates labels
* **classification.py** - Trains a neural network on the classification task and plots the corresponding training curves
* **architectures.py** - Contains the definition of a CNN with periodic padding (imported by classification.py)
* **critical_temperature.py** - Plots the prediction of the neural net for finding the critical temperature
