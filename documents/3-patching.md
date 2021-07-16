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
####**method = "extract_patches"**

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
- `<fov>.npy` file generated from `run_preproc.py`
- `<fov>_NNProbabilities.npy` file generated from `run_segmentation.py -m segmentation`
- `cell_pixel_assignments.pkl` file generated from `run_segmentation.py -m instance_segmentation`
- `cell_positions.pkl` file generated from `run_segmentation.py -m instance_segmentation`

**outputs**
- `stacks_<timepoint>.pkl` 
    - is a dictionary of {key:value} = {`full_path/<tpoint>_<cellid>.h5`: `matrix_dict` }
        - where `matrix_dict` is dictionary of {key:value} = {"mat": <np.array_of_image_patch>}, 
                                                             {"masked_mat": <np.array_of_masked_patch>}
- `cell_positions.pkl` --> rewrites the cell_positions.pkl from the inputs

-------------------------------------------
####**method = "build_trajectories"**

This method 

```text
python run_segmentation.py -m instance_segmentation -c myconfig.yml
```

instance segmentation is done using the clustering method DBSCAN (sklearn.cluster).  The process is as follows:

```text
for each time point
1. filter cells whose probability qualifies it for "foreground".  This is "fg_thr" < 0.3 in the paper.
2. perform DBSCAN clustering with `eps = 10` and `min_samples = 250` (values used in dynamorph paper)
3. position_labels is the output of step 2
4. cell_ids, point_counts is set of unique values from position_labels
5. for each cell_id/point_counts
        define a "mean position" around each cluster
        define a window of 256x256 around that mean
        exclude clusters that have too many outliers outside that window (> 5% of points) 
6.      append (cell_id, mean_pos) to qualifying cells to the `cell_positions` list
7. assign the output of 6 to the dictionary `cell_positions[time_point]`
```

**inputs**
- reads `<fov>.npy`
- reads `<fov>_NNProbabilities.npy`

**outputs**
- writes `cell_positions.pkl`
- writes `cell_pixel_assignments.pkl`

where `cell_positions.pkl` is a dictionary of {key:value} = {timepoint: (microglia-cell-map, non-microglia-cell-map, other-cell-map)}  
and where `<MG or nonMG or other>-cell-map` represents `[ (cell_id, np.array(mean-x-pos, mean-y-pos)), (next_cell_id, np.array(mean-x-pos, mean-y-pos)), ... ]`


where `cell_pixel_assignments.pkl` is a dictionary of {key:value} = {timepoint: (positions, position_labels)}
and where `positions` represents array of (X, Y) coordinates of foreground pixels  
and where `position_labels` represents an array of cell_IDs of those foreground pixels  
