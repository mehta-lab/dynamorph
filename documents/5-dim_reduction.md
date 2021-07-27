# dimensionality reduction

## Purpose

Given vectors representing the learned representation of the images, fit a PCA model and run inference

The relevant CLI is:
```text
python run_dim_reduction.py -m <method> -c <path to config file>
```

where <method> is one of "pca" or "umap"
where <path to config file> is the full path to a .yml configuration file as specified in `.configs/config_example.yml`

--------------------------------------------
#### **method = "pca"**

Fit a PCA model on all latent space representations of the data and on all well prefixes specified in the config

```text
python run_dim_reduction.py -m pca -c myconfig.yml
```

where `myconfig.yml` contains fields under `dim_reduction`:

**important config fields**
```text
dim_reduction:
    input_dirs: <list of full paths to 'raw' subfolder>
    output_dirs: <list of full paths to output PCAed latent spaces>
    weights_dir: <single folder where fitted PCA model will be saved as 'pca_model.pkl'>
    file_name_prefixes: <list of strings that are the prefixes for each `<prefix>_latent_space.pkl` outputted from run_vae.py
    
    fit_model: True or False
```

methodology:
```text
For `fit_model: True`:

1. loops over all directories listed in config's `input_dirs`
2. loops over all prefixes in config's `file_name_prefixes`
3. [aggregate all data]: searches for `<prefix>_latent_space_after.pkl` files in the input dirs and concatenates them in a vector list for subsequent PCA fitting
4. Fitting will write a model `pca_model.pkl` to the config's `weights_dir` directory.
5. Fitting will write a figure `PCA.png` to the config's `weights_dir` directory
6. finally, will loop over all pairs of `input_dirs` and `output_dirs` in the config:
7. will run inference on all individual `<prefix>_latent_space_<suffix>.pkl` in `input_dir` folder, where `suffix='after'` hardcoded.  And where the supplied model is the one generated from step 4 above.
7. output of each inference is `<prefix>_latent_space_after_PCAed.pkl` and saved to each corresponding `output_dir` from 6
```

```text
For `fit_model: False`:
1. loops over all pairs of directories listed in config's `input_dirs` / `output_dirs`
2. loops over all prefixes in config's `file_name_prefixes`
3. assumes the `weights_dir` supplied in the config is a directory, and looks for the `pca_model.pkl` file there.
4. runs inference on `<prefix>_latent_space_<suffix>.pkl` where `suffix=after` is hardcoded.
5. writes the transformed vectors to `<prefix>_latent_space_<suffix>_PCAed.pkl` in the corresponding `output_dir` directory 
```

**inputs**
From `config.dim_reduction.input_dirs`
- `<prefix>_latent_space_after.pkl` files in the input dirs.
- `pca_model.pkl` in the `weights_dir` directory if `fit_model: False`

**outputs**
To `config.dim_reduction.output_dirs` 
- `<prefix>_latent_space_after_PCAed.pkl`

To `config.dim_reduction.weights_dir`, if `fit_model: True`
- `pca_model.pkl`


-------------------------------------------
#### **method = "umap"**

fit a UMAP model on all latent space representations of the data and on all well prefixes specified in the config.

**inputs**
Inputs are the same as PCA except the UMAP method takes only `fit_model: True`.

**outputs**
To `config.dim_reduction.weights_dir`
- `umap_nbr#_a#_b#.pkl`
    - `embedding`: UMAP reduced embeddings
    - `labels`: Class label of embeddings
