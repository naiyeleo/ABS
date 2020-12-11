# ABS


This is repository for paper ABS: Scanning Neural Networks for Back-doors by  Artificial Brain Stimulation. 

## TrojAI competition

This repo also include submission file for TrojAI compeition different rounds. 
Currently, it includes the TrojAi competition round 1 submission. 
Round 1 code is the submission for 20200725T153001 of Perspecta team. This submission has 0.30 cross entropy and 0.91 roc-auc on test set and 0.281 cross entropy loss and 0.92 roc-auc on holdout set. 

## ABS

### Dependences
Python 2, tensoflow=1.12.0, keras=2.2.4, imageio, numpy, pickle, h5py

### File Description

Here we provide binary file to execute ABS in tensorflow and keras.

You can edit `config.json` to change different models and settings for ABS. `models` contain 20 benign models and 21 compromised models. You can edit `config.json` to choose different models.

The seed images for CIFAR-10 dataset is in `cifar_seed_10.pkl` which contains 10 seed images, ABS reads in this file and perform analysis on these data. `cifar_seed_50.pkl` contains 50 seed images and running ABS on more images can increase stability.
The preprossing code of input images is written in `preprocess.py`. ABS calls `cifar.py` and to provide your own preprocess function, just change the code in `cifar.py`.

To run the code, 
`python run_abs.py`
The program will output highest REASR for the model provided in `config.json`.
Triggers with over 80% REASR is shown `imgs` folder. `deltas` and `masks` store the numpy array for such triggers.

Currently, this version of ABS only work on CIFAR-10 dataset and may not support some structure. 
Support for more dataset and structure is coming soon.

Currently, ABS assumes the activation layer and conv/dense layer are seperated, i.e. the conv/dense layers do not have activation function and there is an activation layer after each dense/conv layer. 
Please refer to `reformat_model.py` to see how to seperate activation layers from conv/dense layer.

## Contacts

Yingqi Liu, liu1751@purdue.edu
