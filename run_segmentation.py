# bchhun, {2020-02-21}

from pipeline.segmentation import segmentation, instance_segmentation


# ESS from hulk

# SITES = ['B4-Site_0', 'B4-Site_1',  'B4-Site_2',  'B4-Site_3',  'B4-Site_4', 'B4-Site_5', 'B4-Site_6', 'B4-Site_7', 'B4-Site_8',
#          'B5-Site_0', 'B5-Site_1',  'B5-Site_2',  'B5-Site_3',  'B5-Site_4', 'B5-Site_5', 'B5-Site_6', 'B5-Site_7', 'B5-Site_8',
#          'C3-Site_0', 'C3-Site_1',  'C3-Site_2',  'C3-Site_3',  'C3-Site_4', 'C3-Site_5', 'C3-Site_6', 'C3-Site_7', 'C3-Site_8',
#          'C4-Site_0', 'C4-Site_1',  'C4-Site_2',  'C4-Site_3',  'C4-Site_4', 'C4-Site_5', 'C4-Site_6', 'C4-Site_7', 'C4-Site_8',
#          'C5-Site_0', 'C5-Site_1',  'C5-Site_2',  'C5-Site_3',  'C5-Site_4', 'C5-Site_5', 'C5-Site_6', 'C5-Site_7', 'C5-Site_8']

SITES = ['C5-Site_0']

DATA_PREP = '/gpfs/CompMicro/Hummingbird/Processed/Galina/VAE/data_temp'


def main():

    # loads: 'NNsegmentation/temp_save_unsaturated/final.h5', 'site.npy' (pre=generated using preprocess.py)
    # generates '_NNProbabilities.npy', '
    #           .png',
    #           '_NNpred.png',
    #            '%s-supps' FOLDER
    # prints: "Predicting %d" % t
    segmentation(DATA_PREP, SITES[0])

    # generates 'cell_positions.pkl',
    #           'cell_pixel_assignments.pkl',
    #           'segmentation_%d.png'
    # prints 'Clustering time %d' % timepoint
    instance_segmentation(DATA_PREP, SITES[0])


if __name__ == '__main__':
    main()