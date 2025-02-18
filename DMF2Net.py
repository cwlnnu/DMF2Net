import os
import gc

import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import datasets
from change_detector import ChangeDetector
from image_translation import ImageTranslationNetwork
from change_priors import Degree_matrix, ztz
from config import get_config_kACE
from decorators import image_to_tensorboard
import numpy as np


class Kern_AceNet(ChangeDetector):
    def __init__(self, translation_spec, **kwargs):
        super().__init__(**kwargs)

        self.cycle_lambda = kwargs.get("cycle_lambda", 1)
        self.cross_lambda = kwargs.get("cross_lambda", 1)
        self.recon_lambda = kwargs.get("recon_lambda", 1)
        self.l2_lambda = kwargs.get("l2_lambda", 1e-3)
        self.kernels_lambda = kwargs.get("kernels_lambda", 1)
        self.min_impr = kwargs.get("minimum improvement", 1e-3)
        self.patience = kwargs.get("patience", 10)
        self.crop_factor = kwargs.get("crop_factor", 0.2)

        self._enc_x = ImageTranslationNetwork(
            **translation_spec["enc_X"], name="enc_X", l2_lambda=self.l2_lambda
        )
        self._enc_y = ImageTranslationNetwork(
            **translation_spec["enc_Y"], name="enc_Y", l2_lambda=self.l2_lambda
        )

        self._dec_x = ImageTranslationNetwork(
            **translation_spec["dec_X"], name="dec_X", l2_lambda=self.l2_lambda
        )
        self._dec_y = ImageTranslationNetwork(
            **translation_spec["dec_Y"], name="dec_Y", l2_lambda=self.l2_lambda
        )

        self.loss_object = tf.keras.losses.MeanSquaredError()

        self.train_metrics["cycle_x"] = tf.keras.metrics.Sum(name="cycle_x MSE sum")
        self.train_metrics["cross_x"] = tf.keras.metrics.Sum(name="cross_x MSE sum")
        self.train_metrics["recon_x"] = tf.keras.metrics.Sum(name="recon_x MSE sum")
        self.train_metrics["cycle_y"] = tf.keras.metrics.Sum(name="cycle_y MSE sum")
        self.train_metrics["cross_y"] = tf.keras.metrics.Sum(name="cross_y MSE sum")
        self.train_metrics["recon_y"] = tf.keras.metrics.Sum(name="recon_y MSE sum")
        self.train_metrics["krnls"] = tf.keras.metrics.Sum(name="krnls MSE sum")
        self.train_metrics["l2"] = tf.keras.metrics.Sum(name="l2 MSE sum")
        self.train_metrics["total"] = tf.keras.metrics.Sum(name="total MSE sum")

        self.metrics_history["total"] = []

    def save_all_weights(self):
        self._enc_x.save_weights(self.log_path + "/weights/_enc_x/")
        self._enc_y.save_weights(self.log_path + "/weights/_enc_y/")
        self._dec_x.save_weights(self.log_path + "/weights/_dec_x/")
        self._dec_y.save_weights(self.log_path + "/weights/_dec_y/")

    def load_all_weights(self, folder):
        self._enc_x.load_weights(folder + "/weights/_enc_x/")
        self._enc_y.load_weights(folder + "/weights/_enc_y/")
        self._dec_x.load_weights(folder + "/weights/_dec_x/")
        self._dec_y.load_weights(folder + "/weights/_dec_y/")

    @image_to_tensorboard()
    def enc_x(self, inputs, training=False):
        return self._enc_x(inputs, training)

    @image_to_tensorboard()
    def dec_x(self, inputs, training=False):
        return self._dec_x(inputs, training)

    @image_to_tensorboard()
    def enc_y(self, inputs, training=False):
        return self._enc_y(inputs, training)

    @image_to_tensorboard()
    def dec_y(self, inputs, training=False):
        return self._dec_y(inputs, training)

    def early_stopping_criterion(self):
        self.save_all_weights()
        tf.print(
            "total_loss",
            self.metrics_history["total"][-1],
        )
        return False

    def __call__(self, inputs, training=False):
        x, y = inputs
        tf.debugging.Assert(tf.rank(x) == 4, [x.shape])
        tf.debugging.Assert(tf.rank(y) == 4, [y.shape])

        if training:
            x_code, y_code = self._enc_x(x, training), self._enc_y(y, training)
            x_hat, y_hat = self._dec_x(y_code, training), self._dec_y(x_code, training)
            x_dot, y_dot = (
                self._dec_x(self._enc_y(y_hat, training), training),
                self._dec_y(self._enc_x(x_hat, training), training),
            )
            x_tilde, y_tilde = (
                self._dec_x(x_code, training),
                self._dec_y(y_code, training),
            )
            zx_t_zy = ztz(
                tf.image.central_crop(x_code, self.crop_factor),
                tf.image.central_crop(y_code, self.crop_factor),
            )
            retval = [x_hat, y_hat, x_dot, y_dot, x_tilde, y_tilde, zx_t_zy]

        else:
            x_code, y_code = self.enc_x(x, name="x_code"), self.enc_y(y, name="y_code")
            x_tilde, y_tilde = (
                self.dec_x(x_code, name="x_tilde"),
                self.dec_y(y_code, name="y_tilde"),
            )
            x_hat, y_hat = (
                self.dec_x(y_code, name="x_hat"),
                self.dec_y(x_code, name="y_hat"),
            )
            difference_img = self._difference_img(x_tilde, y_tilde, x_hat, y_hat)
            retval = difference_img

        return retval

    def _train_step(self, x, y, clw):
        with tf.GradientTape() as tape:
            x_hat, y_hat, x_dot, y_dot, x_tilde, y_tilde, ztz = self(
                [x, y], training=True
            )
            Kern = 1.0 - Degree_matrix(
                tf.image.central_crop(x, self.crop_factor),
                tf.image.central_crop(y, self.crop_factor),
            )
            kernels_loss = self.kernels_lambda * self.loss_object(Kern, ztz)
            l2_loss = (
                    sum(self._enc_x.losses)
                    + sum(self._enc_y.losses)
                    + sum(self._dec_x.losses)
                    + sum(self._dec_y.losses)
            )
            cycle_x_loss = self.cycle_lambda * self.loss_object(x, x_dot)
            cross_x_loss = self.cross_lambda * self.loss_object(y, y_hat, clw)
            recon_x_loss = self.recon_lambda * self.loss_object(x, x_tilde)
            cycle_y_loss = self.cycle_lambda * self.loss_object(y, y_dot)
            cross_y_loss = self.cross_lambda * self.loss_object(x, x_hat, clw)
            recon_y_loss = self.recon_lambda * self.loss_object(y, y_tilde)

            total_loss = (
                    cycle_x_loss
                    + cross_x_loss
                    + recon_x_loss
                    + cycle_y_loss
                    + cross_y_loss
                    + recon_y_loss
                    + l2_loss
                    + kernels_loss
            )

            targets_all = (
                    self._enc_x.trainable_variables
                    + self._enc_y.trainable_variables
                    + self._dec_x.trainable_variables
                    + self._dec_y.trainable_variables
            )

            gradients_all = tape.gradient(total_loss, targets_all)

            if self.clipnorm is not None:
                gradients_all, _ = tf.clip_by_global_norm(gradients_all, self.clipnorm)
            self._optimizer_all.apply_gradients(zip(gradients_all, targets_all))

        self.train_metrics["cycle_x"].update_state(cycle_x_loss)
        self.train_metrics["cross_x"].update_state(cross_x_loss)
        self.train_metrics["recon_x"].update_state(recon_x_loss)
        self.train_metrics["cycle_y"].update_state(cycle_y_loss)
        self.train_metrics["cross_y"].update_state(cross_y_loss)
        self.train_metrics["recon_y"].update_state(recon_y_loss)
        self.train_metrics["krnls"].update_state(kernels_loss)
        self.train_metrics["l2"].update_state(l2_loss)
        self.train_metrics["total"].update_state(total_loss)


def test(DATASET="Texas", CONFIG=None):
    if CONFIG is None:
        CONFIG = get_config_kACE(DATASET)

    seed = 46
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.config.run_functions_eagerly(True)

    print(f"Loading {DATASET} data")
    x_im, y_im, EVALUATE, (C_X, C_Y) = datasets.fetch(DATASET, **CONFIG)
    if tf.config.list_physical_devices("GPU") and not CONFIG["debug"]:
        C_CODE = 3
        print("here")
        TRANSLATION_SPEC = {
            "enc_X": {"input_chs": C_X, "filter_spec": [50, 50, C_CODE]},
            "enc_Y": {"input_chs": C_Y, "filter_spec": [50, 50, C_CODE]},
            "dec_X": {"input_chs": C_CODE, "filter_spec": [50, 50, C_X]},
            "dec_Y": {"input_chs": C_CODE, "filter_spec": [50, 50, C_Y]},
        }
    else:
        print("why here?")
        C_CODE = 1
        TRANSLATION_SPEC = {
            "enc_X": {"input_chs": C_X, "filter_spec": [C_CODE]},
            "enc_Y": {"input_chs": C_Y, "filter_spec": [C_CODE]},
            "dec_X": {"input_chs": C_CODE, "filter_spec": [C_X]},
            "dec_Y": {"input_chs": C_CODE, "filter_spec": [C_Y]},
        }
    print("Change Detector Init")
    cd = Kern_AceNet(TRANSLATION_SPEC, **CONFIG)
    print("Training")
    training_time = 0
    cross_loss_weight = tf.expand_dims(tf.zeros(x_im.shape[:-1], dtype=tf.float32), -1)
    for epochs in [50,50,50,50]:
        CONFIG.update(epochs=epochs)
        tr_gen, dtypes, shapes = datasets._training_data_generator(
            x_im[0], y_im[0], cross_loss_weight[0], CONFIG["patch_size"]
        )
        TRAIN = tf.data.Dataset.from_generator(tr_gen, dtypes, shapes)
        TRAIN = TRAIN.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        tr_time, _ = cd.train(TRAIN, evaluation_dataset=EVALUATE, **CONFIG)
        for x, y, _ in EVALUATE.batch(1):
            alpha = cd([x, y])
        cross_loss_weight = 1.0 - alpha
        training_time += tr_time

    cd.load_all_weights(cd.log_path)
    cd.final_evaluate(EVALUATE, **CONFIG)
    metrics = {}
    for key in list(cd.difference_img_metrics.keys()) + list(
        cd.change_map_metrics.keys()
    ):
        metrics[key] = cd.metrics_history[key][-1]
    metrics["F1"] = metrics["TP"] / (
        metrics["TP"] + 0.5 * (metrics["FP"] + metrics["FN"])
    )
    timestamp = cd.timestamp
    epoch = cd.epoch.numpy()
    speed = (epoch, training_time, timestamp)
    del cd
    gc.collect()
    return metrics, speed


if __name__ == "__main__":
    test("Texas")
    # test("California")
    # test("Italy")
    # test("France")
