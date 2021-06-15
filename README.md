# DynaMorph

This is a repo storing codes and scripts for **DynaMorph: learning morphodynamic states of human cells with live imaging and sc-RNAseq**, manuscript of this work can be accessed [here](https://www.biorxiv.org/content/10.1101/2020.07.20.213074v1). Analyzing pipeline of DynaMorph and structure of this repo are introduced below.

![pipeline_fig](graphicalabstract_dynamorph.jpg)

### Table of contents:

- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [DynaMorph Pipeline](#dynamorph-pipeline)
  - [Label-free Imaging](#label-free-imaging)
  - [Cell Segmentation and Tracking](#cell-segmentation-and-tracking)
  - [Latent Representations of Morphology](#latent-representations-of-morphology)
- [Usage](#usage)
- [Citing DynaMorph](#citing-dynamorph)

## Requirements

DynaMorph is developed and tested under Python 3.7, packages below are required.

For u-net segmentation
- [TensorFlow](https://www.tensorflow.org/) ==v.2.1
- [segmentation-models](https://github.com/qubvel/segmentation_models) ==v1.0.1

For preprcoessing, patching, latent-space encoding, latent-space training
- [imageio](https://imageio.github.io/) 
- [tifffile](https://pypi.org/project/tifffile/) 
- [Matplotlib](https://matplotlib.org/) 
- [OpenCV](https://opencv.org/about/) 
- [PyTorch](https://pytorch.org/) 
- [SciPy](https://www.scipy.org/) 
- [scikit-learn](https://scikit-learn.org/) 
- [umap-learn](https://umap-learn.readthedocs.io/en/latest/#)
- [pyyaml](https://pyyaml.org/)
- [h5py](https://docs.h5py.org/en/stable/)
- [POT](https://pythonot.github.io/)


## Getting Started

DynaMorph utilizes a broad set of deep learning and machine learning tools to analyze cell imaging data, [pipeline](https://github.com/czbiohub/dynamorph/tree/master/pipeline) folder contains wrapper methods for easy access to the functionalities of DynaMorph. We also maintained some example scripts `run_preproc.py`, `run_segmentation.py`, `run_patch.py` and `run_VAE.py` to facilitate parallelization of data processing. Check [section](#cell-segmentation-and-tracking) below for functionalities this repo provides.

## DynaMorph Pipeline

DynaMorph starts with raw image files from cell imaging experiments and sequentially applies a set of segmentation and encoding tools. Below we briefly introduced the main processing steps.


### Label-free Imaging, Cell Segmentation, and Tracking

![pipeline_fig](pipeline.jpg)

Starting from any microscopy data (file format is .tif single-page series or multi-page stacks acquired from micro-manager) (panel A), use a segmentation model of your choice to generate semantic segmentation maps (panel C).  In the dynamorph paper, we used Quantitative Label-Free Imaging with Phase and Polarization microscopy to depict cellular Phase and Retardance.

Instance segmentation in this work is based on clustering, related methods can be found in `SingleCellPatch/extract_patches.py`. Cell tracking methods can be found in `SingleCellPatch/generate_trajectories.py`.

To generate segmentation and tracking from scratch, follow steps below:

##### <a name="step1"></a> 1. (optional) prepare training images and labels

##### <a name="step2"></a> 2. (optional) train a classifier to provide per-pixel class probabilities, see scripts in `NNsegmentation/run.py`

##### <a name="step3"></a> 3. prepare inputs as 4-D numpy arrays of shape (n<sub>time frames</sub>, height, width, <sub>channels</sub>), see method `pipeline.preprocess.write_raw_to_npy` for an example

##### <a name="step4"></a> 4. apply trained classifier for semantic segmentation, see method `pipeline.segmentation.segmentation` or `run_segmentation.py` 

##### <a name="step5"></a> 5. use predicted class probabilities for instance segmentation, see method `pipeline.segmentation.instance_segmentation` or `run_segmentation.py` 

### Latent Representations of Morphology
DynaMorph uses VQ-VAE to encode and reconstruct cell image patches, from which latent vectors are used as morphology descriptor. Codes for building and training VAE models are stored in `HiddenStateExtractor/vq_vae.py`.

To extract single cell patches and employ morphology encoding, follow steps below:

##### <a name="step6"></a> 6. extract cell patches based on instance segmentation, see method `pipeline.patch_VAE.extract_patches` or `run_patch.py -m 'extract_patches'` 

##### <a name="step7"></a> 7. extract cell trajectories based on instance segmentation, see method `pipeline.patch_VAE.extract_patches` or `run_patch.py -m 'build_trajectories'` 

##### <a name="step8"></a> 8. Train a VAE for cell patch latent-encoding, see method `run_training.py`

##### <a name="step9"></a> 9. assemble cell patches generated from step 7 to model-compatible datasets, see method `pipeline.patch_VAE.assemble_VAE` or `run_VAE.py -m 'assemble'`

##### <a name="step10"></a> 10. apply trained VAE models on cell patches, see method `pipeline.patch_VAE.process_VAE` or `run_VAE.py -m 'process'` 


## Usage

The dataset accompanying this repository is large and currently available upon request for demonstration. 

Scripts `run_preproc.py`, `run_segmentation.py`, `run_patch.py`, `run_VAE.py` and `run_training.py` provide command line interface, for details please check by using the `-h` option.
Each CLI requires a configuration file (.yaml format) that contains parameters for each stage.  Please see the example: `configs/config_example.yml`

To run the dynamorph pipeline, data should first be assembled into 4-D numpy arrays ([step 3](#step3)). 

Semantic segmentation ([step 4](#step4)) and instance segmentation ([step 5](#step5))):

	python run_segmentation.py -m "segmentation" -c <path-to-your-config-yaml>
	python run_segmentation.py -m "instance_segmentation" -c <path-to-your-config-yaml>

Extract patches from segmentation results ([step 6](#step6)), then connect them into trajectories ([step 7](#step7)):

	python run_patch.py -m "extract_patches" -c <path-to-your-config-yaml>
	python run_patch.py -m "build_trajectories" -c <path-to-your-config-yaml>
	
Train a DNN model (VQ-VAE) to learn a representation of your image data ([step 8](#step8)):

	python run_training.py -c <path-to-your-config-yaml>

Transform image patches into DNN model (VQ-VAE) latent-space by running inference. ([step 9](#step9) and [10](#step10)):

	python run_VAE.py -m "assemble" -c <path-to-your-config-yaml>
	python run_VAE.py -m "process" -c <path-to-your-config-yaml>

Reduce the dimension of latent vectors for visualization by fitting a PCA or UMAP model to the data. For UMAP:

    python run_dim_reduction.py -m "pca" -c <path-to-your-config-yaml>
    
    
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

## Contact Us

If you have any questions regarding this work or code in this repo, feel free to raise an issue or reach out to us through:
- Zhenqin Wu <zqwu@stanford.edu>
- Bryant Chhun <bryant.chhun@czbiohub.org>
- Shalin Mehta <shalin.mehta@czbiohub.org> 
