# NerT-CA: Efficient Dynamic Reconstruction from Sparse-view X-ray Coronary Angiography

## [Project Page] | [Paper]

## About NerT-CA
Three-dimensional (3D) and dynamic 3D+time (4D) reconstruction of coronary arteries from X-ray coronary angiography (CA) has
the potential to improve clinical procedures. However, there are multiple challenges to be addressed, most notably, blood-vessel structure sparsity, poor background and blood vessel distinction, sparse-views,
and intra-scan motion. State-of-the-art reconstruction approaches rely on time-consuming manual or error-prone automatic segmentations, limiting clinical usability. Recently, approaches based on Neural Radiance Fields (NeRF) have shown promise for automatic reconstructions in the sparse-view setting. However, they suffer from long training times due to their dependence on MLP-based representations. We propose NerT-CA, a hybrid approach of Neural and Tensorial representations for accelerated 4D reconstructions with sparse-view CA. Building on top of the previous NeRF-based work, we model the CA scene as a decomposition of low-rank and sparse components, utilizing fast tensorial fields for low-rank static reconstruction and neural fields for dynamic sparse reconstruction. Our approach outperforms previous works in both training time and reconstruction accuracy, yielding reasonable reconstructions from as few as three angiogram views. We validate our approach quantitatively and
qualitatively on representative 4D phantom datasets.

## Method Overview
![Overview of the method](https://github.com/kirstenmaas/NerT-CA/blob/main/imgs/overview.jpg)

## Repository
This repository contains the implementation of the PyTorch models. The code for preprocessing the datasets can be found at the repository of [NeRF-CA](https://github.com/kirstenmaas/NeRF-CA). 

- <b>Models</b>: The models are defined in the /model folder.
- <b>Training</b>: The training code can be found in the folder /train. Our main method can be ran through the <b>run_hybrid.py</b> file, for which the hyperparameters can be defined in the <i>hybrid.txt</i> file.