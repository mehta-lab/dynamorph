# Latent Encoding

## Purpose

Given a **trained** Deep Neural Network, assemble data into a necessary input format for the VQ-VAE network

then run inference on the patch data and generate latent encodings

The relevant CLI is:
```text
python run_VAE.py -m <method> -c <path to config file>
```

where <method> is one of "assemble", "process", or "trajectory_matching" and 
where <path to config file> is the full path to a .yml configuration file as specified in `.configs/config_example.yml`

--------------------------------------------
####**method = "assemble"**

Assemble well data into format needed for DNN inference

```text
python run_VAE.py -m assemble -c myconfig.yml
```

where `myconfig.yml` contains fields under `latent_encoding`:

**important config fields**
```text
latent_encoding:
    weights: <pretrained weights for the DNN of choice>
    patch_type: <either "masked_mat" or "mat">
    network: <name of DNN network.  Currently only "VQ_VAE_z16" is implemented>

```

Methodology:
```text
This pipeline loads data from multiple fovs, adjusts intensities to correct
    for batch effect, and assembles into dataset for model prediction
1. For each fov in the list of fovs in the configuration:
    2. search all `<fov>-supps` for `stacks_<timepoint>.pkl` and gather all those filepaths.
3. "prepare" the result of 2 by using `cv2.resize` to down sample the 256x256 patches into 128x128
    4. these patches are sorted by the unique `tpoint_cellid` name for each cell, the stacked into np.array
5. write the sorted filepaths as `<fov>_file_paths.pkl`
6. write the stack as `<fov>_static_patckes.pkl`

7. using the `<fov>_file_paths.pkl` AND the `cell_traj.pkl`, generate "trajectory relations" used by "matching loss" in vae training:
    8. build a "relations" dictionary of {(cell_id1, cell_id2), n} 
        which can be thought of as a large, sparse MxM matrix, where M is the number of unique cell_ids
        and where n is one of (0, 1, 2) that defines the relationship:
            0 means "different trajectory" or not related
            1 means "non-adjacent frame but same trajectory"
            2 means "adjacent frame and same trajectory"
    9. build a "labels" array which is a 1xM array 
        whose indicies represent the sorted `<fov>_file_paths.pkl` indicies
        and whose values increment by 1 for each new trajectory loaded from all sites.
    9. write the "relations" dictionary as `<fov>_static_patches_relations.pkl`
    10 write the "labels" array as `<fov>_static_patches_labels.pkl`
```

**inputs**
From "<fov>-supps/<fov>" directory
- `stacks_<timepoint>.pkl` file generated from `run_patch.py -m extract_patches`
- `cell-traj.pkl` file generated from `run_patch.py -m build_trajectories`

**outputs**
To "raw" directory
Represents an aggregate of all FOV from within a well ("C5-Site_0", "C5-Site_1", "C5-Site_2", etc... will become "C5_file_paths.pkl")
- `<fov>_file_paths.pkl`
- `<fov>_static_patchkes.pkl`
- `<fov>_static_patches_relations.pkl`
- `<fov>_static_patches_labels.pkl`

-------------------------------------------
####**method = "process"**

Run DNN inference on the assembled data
Loads the `<fov>_static_patches.pkl`, zscores it, then casts as a TensorDataset for Pytorch inference


```text
python run_VAE.py -m process -c myconfig.yml
```

**inputs**
From "raw" directory
- reads `<fov>_file_paths.pkl`
- reads `<fov>_static_patchkes.pkl`

**outputs**
To "raw" directory
- writes `<fov>_latent_space.pkl`
    which is the latent representation of the data before the quantizer
- writes `<fov>_latent_space_after.pkl`
    which is the latent representation of the data after the quantizer


-------------------------------------------
####**method = "trajectory_matching"**

Runs the trajectory matching already executed as part of `method = "assemble"`, but this time indepedent of the assemble

```text
python run_VAE.py -m trajectory_matching -c myconfig.yml
```

**inputs**
From "raw" directory
- reads `<fov>_file_paths.pkl`

From "<fov>-supps/<fov>" directory
- reads `cell_traj.pkl`

**outputs**
To "raw" directory
- writes `<fov>_trajectories.pkl`


