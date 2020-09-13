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

DynaMorph starts with raw image files from cell imaging experiments and sequentially applies a set of segmentation and encoding tools. Codes for the main processing steps are wrapped into methods (saved under `pipeline` folder). Below we briefly introduced their functionalities.

1. `pipeline.preprocess.write_raw_to_npy` performs the transformation of raw multipage tiff files into numpy arrays. Resulting arrays (saved as `.npy` files) contain the full trajectories for each field of view.

    - e.g. `D3-Site_1.npy` contains the raw inputs as a numpy array with shape (100, 2048, 2048, 2): this site has 100 frames and a 2048 by 2048 field of view, two channels are available.

2. `pipeline.segmentation.segmentation` loads the trained U-Net model and applies it to saved image stacks, predicted class probabilities will be saved as `_NNProbabilities.npy` files.

    - e.g. `D3-Site_1_NNProbabilities.npy` contains a numpy array with shape (100, 2048, 2048, 3), the last dimension represents a probability distribution over 3 classes.

3. `pipeline.segmentation.instance_segmentation` performs instance segmentation based on each frame's semantic class mask. A list of cells with their compositions and centroid coordinates will be saved.

4. `pipeline.patch_VAE.extract_patches` extracts single cell patches from image stacks based on instance segmentation results. 

5. `pipeline.patch_VAE.build_trajectories` reads cell properties (of different time steps) and connects them into trajectories.

6. `pipeline.patch_VAE.assemble_VAE` and `pipeline.patch_VAE.process_VAE` load a trained VQ-VAE model and perform cell encoding by extracting latent vectors of single cell patches.

### Label-free Imaging
(for pipeline, can use data acquired from any microscopy source -- file format as .tif)
(in dynamorph paper, we use phase and retardance)
(collect polarization-resolved label-free images using method in <reference to virtual staining paper> )

### Cell Segmentation and Tracking

Scripts under `NNsegmentation` folder contain codes for U-Net based cell segmentation model. `NNsegmentation/run.py` provides an example on the model training procedure. Usage of the segmentation model in the whole pipeline can be found in `pipeline/segmentation.py`.

Instance segmentation in this work is done by clustering, related methods can be found in `SingleCellPatch/extract_patches.py`.

To generate a prediction from scratch, follow steps below:
  1. (optional) prepare training images and labels
  2. (optional) train a classifier, see scripts in `NNsegmentation/run.py`
  3. prepare inputs as numpy arrays
  4. apply trained model for semantic segmentation, see method `pipeline.segmentation.segmentation` or `run_segmentation.py` (uncomment line 71, "segmentation")
  5. use predicted class probabilities for instance segmentation, see method `pipeline.segmentation.instance_segmentation` or `run_segmentation.py` (uncomment line 72, "instance_segmentation")
  6. extract cell patches based on instance segmentation, see method `pipeline.patch_VAE.extract_patches` or `run_patch.py` (uncomment line 44, "extract_patches")  
  7. connect static cell frames to trajectories, see method `pipeline.patch_VAE.build_trajectories` or `run_patch.py` (uncomment line 45, "build_trajectories")

### Latent Representations of Morphology

(6 - train vae) `HiddenStateExtractor.vq_vae`  
(7 - gather dataset for model prediction) `run_VAE` (with "assemble_VAE" uncommented)  
(8 - generate latent space predictions for each cell) `run_VAE` (with "process_VAE" uncommented)
(9 - build trajectories from patches) `run_VAE` (with "trajectory_matching" uncommented)

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
