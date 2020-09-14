# DynaMorph

% Overview %

### Table of contents:

- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [DynaMorph Pipeline](#dynamorph-pipeline)
  - [Label-free Imaging](#label-free-imaging)
  - [Cell Segmentation and Tracking](#cell-segmentation-and-tracking)
  - [Latent Representations of Morphology](#latent-representations-of-morphology)
- [Citing DynaMorph](#citing-dynamorph)

## Requirements

DynaMorph is developed and tested under Python 3.7, packages below are required.

- [imageio](https://imageio.github.io/)
- [Keras](https://keras.io/)
- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)
- [OpenCV](https://opencv.org/about/)
- [PyTorch](https://pytorch.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [segmentation-models](https://github.com/qubvel/segmentation_models)
  - Dynamorph requires segmentation-models v0.2.1
- [TensorFlow](https://www.tensorflow.org/)
- [tifffile](https://pypi.org/project/tifffile/)

## Getting Started

% Overview of the tool %

## DynaMorph Pipeline

### Label-free Imaging
(for pipeline, can use data acquired from any microscopy source -- file format as .tif)
(in dynamorph paper, we use phase and retardance)
(collect polarization-resolved label-free images using method in <reference to virtual staining paper> )

### Cell Segmentation and Tracking

(1 - train a classifier -- unet, rforest, or combination) `NNsegmentation.run.py`  
(2 - semantic segmention using classifier in 1 to get Probabilities) `run_segmentation.py` (with "segmentation" uncommented)  
(3 - instance segmentation to generate cell positions/assignments) `run_segmentation.py` (with "instance_segmentation" uncommented)  
(4 - extract patches from instance segmentation) `run_patch.py` (with "extract_patches" uncommented)  
(5 - extract trajectories from instance segmentation) `run_patch.py` (with "build_trajectories" uncommented)

### Latent Representations of Morphology

(6 - train vae) `HiddenStateExtractor.vq_vae.py`  
(7 - gather dataset for model prediction) `run_VAE.py` (with "assemble_VAE" uncommented)  
(8 - generate latent space predictions for each cell) `run_VAE.py` (with "process_VAE" uncommented)  
(9 - build trajectories from patches) `run_VAE.py` (with "trajectory_matching" uncommented)

(use outputs of 8 and 9 for plots, analysis -- such as those in dynamorph paper figure<letter>)

## Citing DynaMorph

To cite DynaMorph, please use the bibtex entry below:

```
@article{wu2020dynamorph,
  title={DynaMorph: learning morphodynamic states of human cells with live imaging and sc-RNAseq},
  author={Wu, Zhenqin and Chhun, Bryant B and Schmunk, Galina and Kim, Chang and Yeh, Li-Hao and Nowakowski, Tomasz J and Zou, James and Mehta, Shalin B},
  journal={bioRxiv},
  year={2020},
  publisher={Cold Spring Harbor Laboratory}
}
```
