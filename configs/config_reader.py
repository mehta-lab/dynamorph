import yaml
import logging


# def log_warning(msg, *args, **kwargs):
#     """Log message with level WARNING."""
#     # import logging
#
#     logging.getLogger(__name__).warning(msg, *args, **kwargs)


# replicate from aicsimageio logging mechanism
###############################################################################

# modify the logging.ERROR level lower for more info
# CRITICAL
# ERROR
# WARNING
# INFO
# DEBUG
# NOTSET
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)

###############################################################################

# to add a new configuration parameter, simply add the string to the appropriate set here
# FILES = {
#     'raw_dirs',
#     'supp_dirs',
#     'train_dirs',
#     'val_dirs',
#     'weights_dir'
# }

PREPROCESS = {
    'image_dirs',
    'target_dirs',
    'channels',
    'fov',
    'multipage',
    'z_slice',
    'pos_dir'
}

PATCH = {
    'raw_dirs',
    'supp_dirs',
    'channels',
    'fov',
    'gpus',
    'window_size',
    'save_fig',
    'reload',
    'skip_boundary'
}

INFERENCE = {
    'raw_dirs',
    'supp_dirs',
    'model',
    'weights',
    'save_output',
    'gpus',
    'gpu_id',
    'fov',
    'channels',
    'num_classes',
    'window_size',
    'batch_size',
    'num_pred_rnd',
    'seg_val_cat'
}

TRAINING = {
    'raw_dirs',
    'supp_dirs',
    'weights_dirs',
    'model',
    'num_inputs',
    'num_hiddens',
    'num_residual_hiddens',
    'num_residual_layers',
    'num_embeddings',

    'w_a',
    'w_t',
    'channel_mean',
    'channel_std',

    'commitment_cost',
    'alpha',
    'epochs',
    'learning_rate',
    'batch_size',
    'gpus',
    'gpu_id',
    'shuffle_data',
    'transform',
}


class Object:
    pass


class YamlReader(Object):

    def __init__(self):
        self.config = None

        # easy way to assign attributes to each category
        # self.files = Object()
        self.preprocess = Object()
        self.patch = Object()
        self.inference = Object()
        self.training = Object()

    def read_config(self, yml_config):
        with open(yml_config, 'r') as f:
            self.config = yaml.load(f)

            # self._parse_files()
            self._parse_preprocessing()
            self._parse_patch()
            self._parse_inference()
            self._parse_training()

    # def _parse_files(self):
    #     for key, value in self.config['files'].items():
    #         if key in FILES:
    #             setattr(self.files, key, value)
    #         else:
    #             log_warning(f"yaml FILE config field {key} is not recognized")

    def _parse_preprocessing(self):
        for key, value in self.config['preprocess'].items():
            if key in PREPROCESS:
                setattr(self.preprocess, key, value)
            else:
                log.warning(f"yaml PREPROCESS config field {key} is not recognized")

    def _parse_patch(self):
        for key, value in self.config['patch'].items():
            if key in PATCH:
                setattr(self.patch, key, value)
            else:
                log.warning(f"yaml PATCH config field {key} is not recognized")

    def _parse_inference(self):
        for key, value in self.config['inference'].items():
            if key in INFERENCE:
                setattr(self.inference, key, value)
            else:
                log.warning(f"yaml INFERENCE config field {key} is not recognized")

    def _parse_training(self):
        for key, value in self.config['training'].items():
            if key in TRAINING:
                setattr(self.training, key, value)
            else:
                log.warning(f"yaml TRAINING config field {key} is not recognized")




