# ABS


This is repository for paper ABS: Scanning Neural Networks for Back-doors by  Artificial Brain Stimulation. 

## Dependences
Python 2, tensoflow=1.12.0, keras=2.2.4, imageio, numpy, pickle

## ABS
Here we only provide binary file to execute ABS.

You can edit `config.json` to change different models and settings for ABS. `models` contain 20 benign models and 21 compromised models. You can edit `config.json` to choose different models.

The seed images for CIFAR-10 dataset is in `cifar_seed.pkl`, ABS reads in this file and perform analysis on these data.
The preprossing code of input images is written in `preprocess.py`. ABS calls `cifar.py` and to provide your own preprocess function, just change the code in `cifar.py`.

To run the code, 
`python ./abs.pyc`
The program will output highest REASR for the model provided in `config.json`.
Triggers with over 80% REASR is shown `imgs` folder. `deltas` and `masks` store the numpy array for such triggers.

Currently, this version of ABS only work on CIFAR-10 dataset and may not support some structure. 
Support for more dataset and structure is coming soon.

## Contacts

Yingqi Liu, liu1751@purdue.edu
