# Patching

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
```text
latent_encoding:
    weights: <pretrained weights for the DNN of choice>
    network: <name of DNN network.  Currently only "VQ_VAE_z16" is implemented>

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

-------------------------------------------
####**method = "process"**

Run DNN inference on the assembled data

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


