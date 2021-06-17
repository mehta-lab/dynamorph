import yaml
import logging


# replicate from aicsimageio logging mechanism
###############################################################################

# modify the logging.ERROR level lower for more info
# CRITICAL
# ERROR
# WARNING
# INFO
# DEBUG
# NOTSET
#TODO: Save log file to train or supp folders
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)

###############################################################################

# to add a new configuration parameter, simply add the string to the appropriate set here

PREPROCESS = {
    'image_dirs',
    'target_dirs',
    'channels',
    'fov',
    'pos_dir',
    'multipage',
    'z_slice',
}

SEGMENTATION_INFERENCE = {
    'raw_dirs',
    'supp_dirs',
    'validation_dirs',
    'model',
    'weights',
    'gpu_ids',
    'fov',
    'channels',

    'num_classes',
    'window_size',
    'batch_size',
    'num_pred_rnd',
    'seg_val_cat'
}

PATCH = {
    'raw_dirs',
    'supp_dirs',
    'channels',
    'fov',

    'num_cpus',
    'window_size',
    'save_fig',
    'reload',
    'skip_boundary'
}

# change this to "latent encoding" or similar
LATENT_ENCODING = {
    'raw_dirs',
    'supp_dirs',
    'val_dirs',
    'model',
    'weights',
    'save_output',
    'gpu_ids',
    'fov',

    'channels',
    'channel_mean',
    'channel_std',

    'num_classes',
    'num_hiddens',
    'num_residual_hiddens',
    'num_embeddings',
    'num_pred_rnd',
    'commitment_cost',
    'seg_val_cat'
}

DIM_REDUCTION = {
    'input_dirs',
    'output_dirs',
    'file_name_prefixes',
    'weights_dirs',
    'fit_model',
    'conditions'
}

TRAINING = {
    'raw_dirs',
    'supp_dirs',
    'weights_dirs',
    'network',
    'num_inputs',
    'num_hiddens',
    'num_residual_hiddens',
    'num_residual_layers',
    'num_embeddings',
    'weight_matching',
    'margin',
    'w_a',
    'w_t',
    'w_n',
    'channel_mean',
    'channel_std',
    'commitment_cost',
    'n_epochs',
    'learn_rate',
    'batch_size',
    'val_split_ratio',
    'shuffle_data',
    'transform',
    'patience',
    'n_pos_samples',
    'num_workers',
    'gpu_id',
    'start_model_path',
    'retrain',
    'start_epoch',
    'earlystop_metric',
    'model_name',
    'use_mask',
}


class Object:
    pass


class YamlReader(Object):

    def __init__(self):
        self.config = None

        # easy way to assign attributes to each category
        # self.files = Object()
        self.preprocess = Object()
        self.segmentation = Object()
        self.segmentation.inference = Object()
        self.patch = Object()
        self.latent_encoding = Object()
        self.dim_reduction = Object()
        self.training = Object()

    def read_config(self, yml_config):
        with open(yml_config, 'r') as f:
            self.config = yaml.load(f)

            self._parse_preprocessing()
            self._parse_segmentation()
            self._parse_patch()
            self._parse_inference()
            self._parse_dim_reduction()
            self._parse_training()

    def _parse_preprocessing(self):
        for key, value in self.config['preprocess'].items():
            if key in PREPROCESS:
                setattr(self.preprocess, key, value)
            else:
                log.warning(f"yaml PREPROCESS config field {key} is not recognized")

    def _parse_segmentation(self):
        for key, value in self.config['segmentation_inference'].items():
            if key in SEGMENTATION_INFERENCE:
                setattr(self.segmentation.inference, key, value)
            else:
                log.warning(f"yaml SEGMENTATION config field {key} is not recognized")

    def _parse_patch(self):
        for key, value in self.config['patch'].items():
            if key in PATCH:
                setattr(self.patch, key, value)
            else:
                log.warning(f"yaml PATCH config field {key} is not recognized")

    def _parse_inference(self):
        for key, value in self.config['latent_encoding'].items():
            if key in LATENT_ENCODING:
                setattr(self.latent_encoding, key, value)
            else:
                log.warning(f"yaml LATENT_ENCODING config field {key} is not recognized")

    def _parse_dim_reduction(self):
        for key, value in self.config['dim_reduction'].items():
            if key in DIM_REDUCTION:
                setattr(self.dim_reduction, key, value)
            else:
                log.warning(f"yaml DIM REDUCTION config field {key} is not recognized")

    def _parse_training(self):
        for key, value in self.config['training'].items():
            if key in TRAINING:
                setattr(self.training, key, value)
            else:
                log.warning(f"yaml TRAINING config field {key} is not recognized")




