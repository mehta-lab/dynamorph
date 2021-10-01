import pickle
import os

from analysis.encoding_metrics import size, peak_phase, peak_retardance, aspect_ratio, aspect_ratio_no_rotation


# ===================================================================
# USAGE
# ===================================================================


"""
SingleCellMetrics Class is meant to be an extendable class that intends to:
- loop over sites once
- generate output dictionaries for dynamic use
- write output dictionaries to intermediate files for later use

Usage:
c = SingleCellMetrics(<path to supp folder including well-supps>, 
                sites=["C2-Site_0", "C2-Site_1", etc..],
                metrics=['size'],
                out_path=<path to folder to write intermediate files> -- optional
                )
c.add_metrics(['aspect_ratio', 'peak_phase'])
a = c.compute()     # will also write to out_path if supplied
dict_of_single_cell_patch_aspect_ratios = a[0]
dict_of_single_cell_patch_peak_phase = a[1]

"""

# ===================================================================
# MAP NAMES TO FUNCTIONS
# ===================================================================

"""
Dictionary of {<user supplied string/name> : metric function name imported from encoding_metrics.py}
For each metric implemented in analysis.encoding_metrics.py, describe them here.
    also define an output filename if you wish to write to file.
    
"""
NAMESPACE_METRIC_MAPPING = {
    'size': size,
    'peak_phase': peak_phase,
    'peak_retardance': peak_retardance,
    'aspect_ratio': aspect_ratio,
    'aspect_ratio_no_rotation': aspect_ratio_no_rotation
    }

"""
Dictionary of {<user supplied string/name> : output pickled filename}
"""
NAMESPACE_FILENAME_MAPPING = {
    'size': "EncodedSizes.pkl",
    'peak_phase': "EncodedPeakPhase.pkl",
    'peak_retardance': "EncodedPeakRetardance.pkl",
    'aspect_ratio': "EncodedAspectRatio.pkl",
    'aspect_ratio_no_rotation': "EncodedAspectRatioNoRotation.pkl"
    }

# ===================================================================
# END METRIC DEFINITIONS
# ===================================================================


class SingleCellMetrics:

    metric_mapping = NAMESPACE_METRIC_MAPPING

    filename_mapping = NAMESPACE_FILENAME_MAPPING

    def __init__(self, supp_folder, sites='all', metrics=['size', 'peak_phase'], out_path=None):
        """
        given a path to a supplementary folder:
            - find all "stacks_<timepoint>.pkl" within subfolders
            - extract all "masked_mat" binary masks for each "cell id"
                create a
            - write the per-cell-id summary to a dictionary called EncodedSizes.pkl

        :param supp_folder : (str)
            path to supplementary folder generated through dynamorph pipeline
            this should be the subfolder ending with "<well>-supps"
        :param sites : (list)
            FOV subset that will be processed.  Default = 'all"
        :param metrics: (list)
            list of strings defining
        :param out_path : (str)
            destination path to write encoded sizes
        """

        self.sites = [f for f in supp_folder if "Site" in f] if sites == "all" else sites
        self.metrics_list = self._check_metrics(metrics)
        self.out_path = out_path
        self.supp_folder = supp_folder

    @property
    def metrics(self):
        return self.metrics_list

    def add_metrics(self, value):
        if type(value) is list:
            m = self._check_metrics(value)
            self.metrics_list.extend(m)

        if type(value) is str:
            if value not in self.metric_mapping:
                raise NotImplementedError(f"supplied metric {value} is not implemented")
        else:
            self.metrics_list.append(value)

    def remove_metrics(self, value):
        try:
            if type(value) is list:
                for v in value:
                    self.metrics_list.remove(v)

            if type(value) is str:
                self.metrics_list.remove(value)
        except Exception as ex:
            print(f"exception raised while trying to remove metrics: {ex}")

    def _check_metrics(self, metrics_: list):
        for m in metrics_:
            if m not in self.metric_mapping:
                print(f"metric {m} is not implemented.  Removing from outputs")
                metrics_.remove(m)
        if bool(metrics_) is False:
            raise NotImplementedError("no supplied metrics have been implemented")
        return metrics_

    def _assign_metric_attribute(self):
        # assign metric name as class attribute
        for metric in self.metrics:
            if metric in self.metric_mapping:
                # check that attribute does not exist so we don't overwrite it
                if not hasattr(self, metric):
                    self.__setattr__(metric, dict())

    def _feed_stack(self):
        """
        iterate over all sites within the folder
        ** for now this is a generator but could easily return a list.  This list could be enormous depending on the dataset.

        :return: (dict)
            a single FOV's timepoint's patch mapping,  Can be multiple cell Ids per dict

            {key: value} = {cell-id as filepath : {'mat':np.array, 'masked_mat':np.array}}
        """
        for site in self.sites:
            print(f"generating cell sizes for site {site}")
            p = os.path.join(self.supp_folder, site)
            site_stacks = sorted([os.path.join(self.supp_folder, site, f) for f in os.listdir(p) if "stacks" in f])
            for i, stack in enumerate(site_stacks):

                if i % 25 == 0:
                    print(f"\tloading stack {stack}")

                stack_timepoint = pickle.load(open(os.path.join(p, stack), 'rb'))
                yield stack_timepoint

    def _compute_metric(self, metric, fn, timepoint_stack):
        # reference dictionary map to metric function
        if metric in self.metric_mapping:
            func = self.metric_mapping[metric]
            return func(timepoint_stack, fn)

    def compute(self):
        for tpstk in self._feed_stack():
            for fn in tpstk.keys():
                for metric in self.metrics:
                    value = self._compute_metric(metric, fn, tpstk)
                    self.__getattribute__(metric)[fn] = value

        if self.out_path:
            self._write_metric()

        return [self.__getattribute__(m) for m in self.metrics]

    # should this func receive a type (sizes, density etc..) or should it
    def _write_metric(self):
        for metric in self.metrics:
            path = os.path.join(self.out_path, self.filename_mapping[metric])
            attr = self.__getattribute__(metric)
            with open(path, 'wb') as fw:
                pickle.dump(attr, fw)
