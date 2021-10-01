
from HiddenStateExtractor.cv2_feature import get_angle_apr, get_aspect_ratio_no_rotation


# ===================================================================
# DEFINE METRIC FUNCTIONS HERE
# ===================================================================


def size(timepoint_stack, cellid):
    return timepoint_stack[cellid]['masked_mat'][2].sum()


def peak_phase(timepoint_stack, cellid):
    return timepoint_stack[cellid]['mat'][0].max()


def peak_retardance(timepoint_stack, cellid):
    return timepoint_stack[cellid]['mat'][1].max()


def aspect_ratio(timepoint_stack, cellid):
    """
    computes aspect ratio of masked patch accounting for rotation and returns bounding box width, height and angle

    :param timepoint_stack:
    :param cellid:
    :return:
    """
    msk = timepoint_stack[cellid]['masked_mat'][2]
    width, height, angle = get_angle_apr(msk)
    return width, height, angle


def aspect_ratio_no_rotation(timepoint_stack, cellid):
    """
    computes aspect ratio of masked patch without accounting for rotation and returns bounding box width, height

    :param timepoint_stack:
    :param cellid:
    :return:
    """
    msk = timepoint_stack[cellid]['masked_mat'][2]
    width, height = get_aspect_ratio_no_rotation(msk)
    return width, height


