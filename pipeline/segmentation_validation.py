import pickle
import numpy as np
import os
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import skimage.io as io


def find_rim(cell_positions):
    masks = set(tuple(r) for r in cell_positions)
    inner_masks = set((r[0]-1, r[1]) for r in masks) & \
                set((r[0]+1, r[1]) for r in masks) & \
                set((r[0], r[1]-1) for r in masks) & \
                set((r[0], r[1]+1) for r in masks)
    edge_positions = np.array(list(masks - inner_masks))
    return edge_positions


def drawContour(m, s, c, RGB):
    """Draw edges of contour 'c' from segmented image 's' onto 'm' in colour 'RGB'"""
    # Fill contour "c" with white, make all else black
    #     thisContour = s.point(lambda p:p==c and 255)
    # DEBUG: thisContour.save(f"interim{c}.png")
    #     thisContour = s.point(lambda p:p==c and 255)
    thisContour = s.point(lambda x: 255 if x > 30 else False)

    # Find edges of this contour and make into Numpy array
    thisEdges = thisContour.filter(ImageFilter.FIND_EDGES)
    thisEdgesN = np.array(thisEdges)

    # Paint locations of found edges in color "RGB" onto "dim_reduction"
    m[np.nonzero(thisEdgesN)] = RGB
    return m


def rescale_plot(arr1, filename, size=(1108, 1108), dpi=500):
    plt.clf()

    if type(arr1) == np.ndarray:
    #         arr1 = auto_contrast_slice(arr1)
        im = Image.fromarray(arr1)
    else:
        im = arr1
    im = im.resize(size)

    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    plt.imshow(np.array(im))
    if type(arr1) == np.ndarray:
        plt.clim(0, 0.9 * arr1.max())
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=dpi)


# %%
def load_and_plot(img_rgb, img_grey, output):
    phase = Image.open(img_rgb).convert('L').convert('RGB')
    segment = Image.open(img_grey).convert('L')

    phaseN = np.array(phase)
    phaseN = drawContour(phaseN, segment, 0, (255, 0, 0))
    Image.fromarray(phaseN).save(output)


def segmentation_validation_michael(paths, gpu_id, category):
    """

    :param paths:
    :param category: str
        only "mg", "nonmg", "both", "unfiltered"
    :param gpu_id:
    :return:
    """

    temp_folder, supp_folder, target, sites = paths[0], paths[1], paths[2], paths[3]

    if "NOVEMBER" in temp_folder:
        date = "NOVEMBER"
    elif "JANUARY_FAST" in temp_folder:
        date = "JANUARY_FAST"
    else:
        date = "JANUARY"

    for site in sites:
        print(f"building full frame validation for {site} from {temp_folder}")

        stack_path = os.path.join(temp_folder + '/' + site + '.npy')
        raw_input_stack = np.load(stack_path)

        NN_predictions_stack = np.load(os.path.join(temp_folder, '%s_NNProbabilities.npy' % site))
        cell_pixels = pickle.load(open(os.path.join(supp_folder, f"{site[0:2]}-supps/{site}/cell_pixel_assignments.pkl"), 'rb'))

        # option to include filtered positions
        filtered_positions = pickle.load(
            open(supp_folder + f"/{site[0:2]}-supps/{site}/cell_positions.pkl", 'rb'))

        stack = []
        for t_point in range(len(raw_input_stack)):
            print(f"\tprocessing t {t_point}")
            mat = raw_input_stack[t_point, :, :, 0]
            mat = np.stack([mat] * 3, 2)

            # this block represents rendering of FILTERED MG and nonMG cells
            (mg_cell_positions, non_mg_cell_positions, other_cells) = filtered_positions[t_point]

            # this block represents rendering of MG and nonMG cells, but NOT filtered by size.
            positions, inds = cell_pixels[t_point]

            if 'unfiltered' in category:
                for cell_ind in np.unique(inds):
                    new_mat = _append_segmentation(positions, inds, cell_ind, NN_predictions_stack, t_point, mat)
                    if new_mat is not None:
                        mat = new_mat
            elif 'both' in category:
                if mg_cell_positions is None:
                    if non_mg_cell_positions is None:
                        continue
                    else:
                        ids = non_mg_cell_positions
                else:
                    if non_mg_cell_positions is None:
                        ids = mg_cell_positions
                    else:
                        # ForkedPdb().set_trace()
                        ids = [i for i, _ in mg_cell_positions+non_mg_cell_positions]
            
                for both_cell_id in ids:
                    new_mat = _append_segmentation(positions, inds, both_cell_id, NN_predictions_stack, t_point, mat)
                    if new_mat is not None:
                        mat = new_mat

            elif 'mg' in category:
                # ForkedPdb().set_trace()
                ids = [i for i, _ in mg_cell_positions]
                for mg_cell_id in ids:
                    new_mat = _append_segmentation(positions, inds, mg_cell_id, NN_predictions_stack, t_point, mat)
                    if new_mat is not None:
                        mat = new_mat
            elif 'nonmg' in category:
                ids = [i for i, _ in non_mg_cell_positions]
                for non_mg_cell_id in ids:
                    new_mat = _append_segmentation(positions, inds, non_mg_cell_id, NN_predictions_stack, t_point, mat)
                    if new_mat is not None:
                        mat = new_mat
            else:
                raise NotImplementedError(f"rendering category of type {category} is not impemented")
            
            stack.append(mat)

        # tifffile.imwrite(target+'/'+f'{date}_{site}_predictions.tiff', np.stack(stack, 0))
        # np.save(target+'/'+f'{date}_{site}_predictions.npy', np.stack(stack, 0))

        # using skimage.io to access tifffile on IBM machines
        # ForkedPdb().set_trace()
        io.imsave(target+'/'+f'{date}_{site}_{gpu_id}_predictions.tif',
                  np.stack(stack, 0).astype("uint16"),
                  plugin='tifffile')


def _append_segmentation(positions_, inds_, cell_id_, NN_predictions_stack_, t_point_, output_mat_):
    """
    adds boundary positions for a supplied cell
    :param positions_:
    :param inds_:
    :param cell_id_:
    :param NN_predictions_stack_:
    :param t_point_:
    :param output_mat_:
    :return:
    """
    if cell_id_ < 0:
        return None

    cell_positions = positions_[np.where(inds_ == cell_id_)]

    outer_rim = find_rim(cell_positions)
    mask_identities = NN_predictions_stack_[t_point_][(cell_positions[:, 0], cell_positions[:, 1])].mean(0)
    if mask_identities[1] > mask_identities[2]:
        c = 'b'
        output_mat_[(outer_rim[:, 0], outer_rim[:, 1])] = np.array([0, 65535, 0]).reshape((1, 3))
    else:
        c = 'r'
        output_mat_[(outer_rim[:, 0], outer_rim[:, 1])] = np.array([65535, 0, 0]).reshape((1, 3))
    return output_mat_


def segmentation_validation_bryant(paths, id):
  """
  this approach uses the outputted .png segmentations (per frame) and stitches it back with the raw data using PIL

  :param paths:
  :return:
  """

  temp_folder, supp_folder, target, sites = paths[0], paths[1], paths[2], paths[3]

  for site in sites:
    print(f"building full frame validation for {site} from {temp_folder}")

    stack_path = os.path.join(temp_folder + '/' + site + '.npy')
    segmentations_png_path = os.path.join(supp_folder + f"/{site[0:2]}-supps/{site}")

    raw_input_stack = np.load(stack_path)

    for tp in range(len(raw_input_stack)):
      seg = Image.open(segmentations_png_path + os.sep + f'segmentation_{tp}.png').convert('L')

      # site[t,:,:,0] is phase channel
      rescale_plot(raw_input_stack[tp, :, :, 0], target + f"/temp_phase_{id}.png")
      rescale_plot(seg, target + f"/temp_seg_{id}.png")

      if "NOVEMBER" in temp_folder:
        date = "NOVEMBER"
      else:
        date = "JAN_FAST"

      if not os.path.exists(target+'/'+date):
        os.makedirs(target+'/'+date)

      load_and_plot(target + f"/temp_phase_{id}.png",
                    target + f"/temp_seg_{id}.png",
                    target + f"/{date}/{date}_{site}_{tp}.png")


def segmentation_validation_to_tiff(paths):
    """
    paths is a tuple of:
    (target folder, date, sites)

    target folder is the EXPERIMENT folder (not subfolder)

    :param paths:
    :return:
    """
    import tifffile as tf
    import imageio as io

    target, date, sites = paths[0], paths[1], paths[2]

    for site in sites:
        png_path = f"{target}/{date}/"

        matched = [file for file in os.listdir(png_path) if f"{date}_{site}" in file]
        smatched = sorted(matched)

        ref = io.imread(png_path+'/'+smatched[0])
        x, y, c = ref.shape
        output = np.empty(shape=(len(smatched), x, y, c))
        for idx, path in enumerate(smatched):
            frame = io.imread(png_path+'/'+path)
            output[idx] = frame

        io.mimwrite(png_path+f'/{date}_{site}_composite.tif', output)

import sys
import pdb


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
