# ABS


This is repository for paper ABS: Scanning Neural Networks for Back-doors by  Artificial Brain Stimulation. 

The repo contains two parts. The source code of ABS pytorch version used in TrojAI competition and the source code of ABS tensorflow+keras version.

## The source code of ABS pytorch version 

This repo include source code of ABS pytorch version. 

The source code of ABS pytorch version for TrojAI competition round 1-4 can be accessed at https://github.com/naiyeleo/ABS/blob/master/TrojAI_competition/
The ABS for TrojAI competition is enchanced for better performance. For example, during trigger reverse engineering besides the loss for stimulating compromised neuron we also include the loss that enlarge the target label's logits value.  


The source code of ABS pytorch version for TrojAI competition round 5 can be accessed at https://github.com/naiyeleo/ABS/blob/master/TrojAI_competition/round5
Round 5 code is the submission for 20210316T161002 of Perspecta-PurdueRutgers team This submission has 0.32 cross entropy and 0.93 roc-auc and 0.26 cross entropy and 0.95 roc-auc on holdout set.


## The source code of ABS tensorflow+keras version

This repo also include the source code of ABS tensorflow+keras version

### Dependences
Python 2, tensoflow=1.12.0, keras=2.2.4, imageio, numpy, pickle, h5py

### File Description

You can edit `config.json` to change different models and settings for ABS. `models` contain 20 benign models and 21 compromised models. You can edit `config.json` to choose different models.

The seed images for CIFAR-10 dataset is in `cifar_seed_10.pkl` which contains 10 seed images, ABS reads in this file and perform analysis on these data. `cifar_seed_50.pkl` contains 50 seed images and running ABS on more images can increase stability.
The preprossing code of input images is written in `preprocess.py`. ABS calls `cifar.py` and to provide your own preprocess function, just change the code in `cifar.py`.

To run the code, 
`python abs.py`
The program will output highest REASR for the model provided in `config.json`.
Triggers with over 80% REASR is shown `imgs` folder. `deltas` and `masks` store the numpy array for such triggers.

Currently, this version of ABS only work on CIFAR-10 dataset and may not support some structure. 
You can change the `abs.py` to work your structure.

Currently, ABS assumes the activation layer and conv/dense layer are seperated, i.e. the conv/dense layers do not have activation function and there is an activation layer after each dense/conv layer. 
Please refer to `reformat_model.py` to see how to seperate activation layers from conv/dense layer.

## Contacts

Yingqi Liu, liu1751@purdue.edu
