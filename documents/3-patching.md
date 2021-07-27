# Patching

## Purpose

This document describes the process of extracting patches of single cells identified from the segmentation step.  It uses the metadata generated from `segmentation` and `instance_segmentation`

The relevant CLI is:
```text
python run_patch.py -m <method> -c <path to config file>
```

where <method> is one of "extract_patches" or "build_trajectories" and 
where <path to config file> is the full path to a .yml configuration file as specified in `.configs/config_example.yml`

--------------------------------------------
#### **method = "extract_patches"**

This method generates a `stacks_<timepoint>.pkl` file and 

```text
python run_patch.py -m extract_patches -c myconfig.yml
```

where `myconfig.yml` contains fields under `patch`:
```text
patch:
    channels: [0,1]
    fov: ['C4-Site_5', 'C4-Site_1', etc...]
```

**inputs**
From "raw" directory
- `<fov>.npy` file generated from `run_preproc.py`
- `<fov>_NNProbabilities.npy` file generated from `run_segmentation.py -m segmentation`

From "<fov>-supps/<fov>" directory
- `cell_pixel_assignments.pkl` file generated from `run_segmentation.py -m instance_segmentation`
- `cell_positions.pkl` file generated from `run_segmentation.py -m instance_segmentation`

**outputs**
To "<fov>-supps/<fov>" directory
- `stacks_<timepoint>.pkl` 
    - is a dictionary of {key:value} = {`full_path/<tpoint>_<cellid>.h5`: `matrix_dict` }
        - where `matrix_dict` is dictionary of {key:value} = {"mat": <np.array_of_image_patch>}, 
                                                             {"masked_mat": <np.array_of_masked_patch>}
- `cell_positions.pkl` --> rewrites the cell_positions.pkl from the inputs

-------------------------------------------
#### **method = "build_trajectories"**

This method builds a `cell_traj.pkl` file that describes cell motion between frames

```text
python run_patch.py -m build_trajectories -c myconfig.yml
```

Methodology
```text
for each fov's supplementary folder
1. gather the cell centroid positions and sizes from the `cell_positions.pkl` and the `cell_pixel_assignments.pkl` files.
2. for each time point `T`
    3. gather all cell positions and sizes at `T`, as well as the cells for timepoint `T+1`
    4. generate pairs of "matched" cells by using `scipy.optimize.linear_sum_assignment`
        whose cost matrix is based on a 100 pixel distance cutoff
5. with all pairwise "matchings" for all timepoints, generate full trajectories using:
    - "Robust single-particle tracking in live-cell time-lapse sequences" (https://www.nature.com/articles/nmeth.1237)
        - an approach to model gaps, splits and merges of objects over time.
```

**inputs**
From "raw" directory
- reads `<fov>.npy`

From "<fov>-supps" subdirectory
- reads `cell_positions.pkl`
- reads `cell_pixel_assignments.pkl`

**outputs**
To "<fov>-supps" directory
- writes `cell_traj.pkl`
    - is a list of `[cell_trajectories, cell_trajectory_positions]`
        where cell_trajectories is a dictionary of {t_point: cell_ID}
        where cell_trajectory_positions is a dictionary of {t_point: cell_center_position}
