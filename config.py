from filtering import decorated_median_filter, decorated_gaussian_filter
import tensorflow as tf
from datetime import datetime


def get_config(dataset_name, debug=False):
    CONFIG = {
        "debug": debug,  # debugging flag
        "clipnorm": 1,  # threshold for gradient clipping
        "learning_rate": 1e-4,  # initial learning rate for ExponentialDecay
        "l2_lambda": 1e-6,  # l2 regularizer for the weights
        "save_images": True,  # bool, wheter to store images after training
        "channel_x": [1, 2, 3],
        "channel_y": [3, 4, 7],
        "filter_": decorated_median_filter("z_median_filtered_diff"),
        "final_filter": decorated_gaussian_filter("z_gaussian_filtered_diff"),
        "patience": 10,  # epochs after which training stops if kernel loss does not improve much
        "minimum improvement": 1e-3,  # minimum improvement in the kernels loss
    }

    if tf.config.list_physical_devices("GPU") and not debug:
        CONFIG.update(
            {
                "logdir": f"logs/{dataset_name}/"
                + datetime.now().strftime("%Y%m%d-%H%M%S"),
                "list_epochs": [50, 50, 50, 50],  # number of training epochs
                "batches": 10,  # number of batches per epoch
                "batch_size": 10,  # number of samples per batch
                "patch_size": 100,  # size of patches extracted for training
                "affinity_batch_size": 500,  # batch size for prior computation
                "affinity_patch_size": 20,  # patch size for prior computation
                "affinity_stride": 5,  # stride for prior computation
            }
        )
        CONFIG.update(crop_factor=24 / CONFIG["patch_size"])
    else:
        CONFIG.update(
            {
                "logdir": f"logs/{dataset_name}/debug/"
                + datetime.now().strftime("%Y%m%d-%H%M%S"),
                "list_epochs": [2, 2, 1],  # number of training epochs
                "batches": 2,  # number of batches per epoch
                "batch_size": 2,  # number of samples per batch
                "patch_size": 10,  # square size of patches extracted for training
                "affinity_batch_size": 10,  # batch size for prior computation
                "affinity_patch_size": 20,  # patch size for prior computation
                "affinity_stride": 5,  # stride for prior computation
            }
        )
        CONFIG.update(crop_factor=0.2)
    CONFIG.update(epochs=sum(CONFIG["list_epochs"]))
    if CONFIG["epochs"] > 20:  # Too many images on Tensorboard, massive logfiles
        freq = CONFIG["epochs"] // 10  # Ten (or nine) copies per image
        CONFIG.update({"evaluation_frequency": freq})
    return CONFIG


def get_config_kACE(dataset_name, debug=False):
    CONFIG = get_config(dataset_name, debug=debug)
    CONFIG.update(
        {
            "cycle_lambda": 1,  # weight for cyclic loss term
            "cross_lambda": 1,  # weight for cross loss term
            "recon_lambda": 1,  # weight for the reconstruction term
            "kernels_lambda": 1,  # weight for the kernel term
        }
    )
    return CONFIG