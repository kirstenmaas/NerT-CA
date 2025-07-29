# NerT-CA: Efficient Dynamic Reconstruction from Sparse-view X-ray Coronary Angiography

## [Project Page] | [Paper]

## About NerT-CA
NerT-CA is a hybrid approach of Neural and Tensorial representations for accelerated 4D reconstructions with sparse-view CA. Building on top of the previous NeRF-based work [NeRF-CA](https://github.com/kirstenmaas/NeRF-CA), we model the CA scene as a decomposition of low-rank and sparse components, utilizing fast tensorial fields for low-rank static reconstruction and neural fields for dynamic sparse reconstruction. Our approach outperforms previous works in both training time and reconstruction accuracy, yielding reasonable reconstructions from as few as three angiogram views. We validate our approach quantitatively and
qualitatively on representative 4D phantom datasets.

## Method Overview
![Overview of the method](https://github.com/kirstenmaas/NerT-CA/blob/main/imgs/overview.jpg)

## Repository
This repository contains the implementation of the PyTorch models. The code for preprocessing the datasets can be found at the repository of [NeRF-CA](https://github.com/kirstenmaas/NeRF-CA). 

- <b>Models</b>: The models are defined in the /model folder.
- <b>Training</b>: The training code can be found in the folder /temporal. Our main method can be ran through the <b>run_hybrid.py</b> file, for which the hyperparameters can be defined in the <i>hybrid.txt</i> file.
