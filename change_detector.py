import os.path
import tensorflow as tf

from datetime import datetime
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tqdm import trange

from filtering import threshold_otsu
from decorators import image_to_tensorboard, timed
from tensorflow_addons.metrics import CohenKappa
from config import get_config
import datasets
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pdb import set_trace as bp


class ChangeDetector:
    def __init__(self, **kwargs):
        learning_rate = kwargs.get("learning_rate", 1e-4)
        lr_all = ExponentialDecay(
            learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True
        )
        self._optimizer_all = tf.keras.optimizers.Adam(lr_all)
        lr_k = ExponentialDecay(
            learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True
        )
        self._optimizer_k = tf.keras.optimizers.Adam(lr_k)
        self.clipnorm = kwargs.get("clipnorm", None)

        self.train_metrics = {}
        self.difference_img_metrics = {"AUC": tf.keras.metrics.AUC()}
        self.change_map_metrics = {
            "ACC": tf.keras.metrics.Accuracy(),
            "Kappa": CohenKappa(num_classes=2),
            "TP": tf.keras.metrics.TruePositives(),
            "TN": tf.keras.metrics.TrueNegatives(),
            "FP": tf.keras.metrics.FalsePositives(),
            "FN": tf.keras.metrics.FalseNegatives(),
        }
        assert not set(self.difference_img_metrics) & set(self.change_map_metrics)
        self.metrics_history = {
            **{key: [] for key in self.change_map_metrics.keys()},
            **{key: [] for key in self.difference_img_metrics.keys()},
        }

        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.channels = {"x": kwargs.get("channel_x"), "y": kwargs.get("channel_y")}

        self._save_images = tf.Variable(False, trainable=False)

        logdir = kwargs.get("logdir", None)
        if logdir is not None:
            self.log_path = logdir
            self.tb_writer = tf.summary.create_file_writer(self.log_path)
            self._image_dir = tf.constant(os.path.join(self.log_path, "images"))
        else:
            self.tb_writer = tf.summary.create_noop_writer()

        self.evaluation_frequency = tf.constant(
            kwargs.get("evaluation_frequency", 1), dtype=tf.int64
        )
        self.epoch = tf.Variable(0, dtype=tf.int64)
        self.stopping = tf.Variable(0, dtype=tf.int32)

    @image_to_tensorboard(static_name=None)
    # @tf.function
    def _domain_difference_img(
        self, original, transformed, bandwidth=tf.constant(3, dtype=tf.float32)
    ):
        d = tf.norm(original - transformed, ord=2, axis=-1)
        threshold = tf.math.reduce_mean(d) + bandwidth * tf.math.reduce_std(d)
        d = tf.where(d < threshold, d, threshold)

        return tf.expand_dims(d / tf.reduce_max(d), -1)

    # @tf.function
    def _difference_img(self, x, y, x_hat, y_hat):
        assert x.shape[0] == y.shape[0] == 1, "Can not handle batch size > 1"

        d_x = self._domain_difference_img(x, x_hat, name="x_ut_diff")
        d_y = self._domain_difference_img(y, y_hat, name="y_ut_diff")

        c_x, c_y = x.shape[-1], y.shape[-1]
        d = (c_y * d_x + c_x * d_y) / (c_x + c_y)

        return d

    # @tf.function
    def _change_map(self, difference_img):
        tmp = tf.cast(difference_img * 255, tf.int32)
        threshold = threshold_otsu(tmp) / 255

        return difference_img >= threshold

    @image_to_tensorboard(static_name="z_Confusion_map")
    # @tf.function
    def _confusion_map(self, target_change_map, change_map):
        conf_map = tf.concat(
            [
                target_change_map,
                change_map,
                tf.math.logical_and(target_change_map, change_map),
            ],
            axis=-1,
            name="confusion map",
        )

        return tf.cast(conf_map, tf.float32)

    def early_stopping_criterion(self):
        return False

    @timed
    def train(
        self,
        training_dataset,
        epochs,
        batches,
        batch_size,
        evaluation_dataset=None,
        filter_=None,
        final_filter=None,
        **kwargs,
    ):
        self.stopping.assign(0)
        for epoch in trange(self.epoch.numpy() + 1, self.epoch.numpy() + epochs + 1):
            self.epoch.assign(epoch)
            tf.summary.experimental.set_step(self.epoch)

            for i, batch in zip(range(batches), training_dataset.batch(batch_size)):
                self._train_step(*batch)

            with tf.device("cpu:0"):
                with self.tb_writer.as_default():
                    for name, metric in self.train_metrics.items():
                        tf.summary.scalar(name, metric.result())
                        try:
                            self.metrics_history[name].append(metric.result().numpy())
                        except KeyError as e:
                            pass
                        metric.reset_states()

            if evaluation_dataset is not None:
                for eval_data in evaluation_dataset.batch(1):
                    ev_res = self.evaluate(*eval_data, filter_)

            tf.summary.flush(self.tb_writer)
            if self.early_stopping_criterion():
                break

        return self.epoch

    def evaluate(self, x, y, target_change_map, filter_=None):
        difference_img = self((x, y))
        if filter_ is not None:
            difference_img = filter_(self, x, y, difference_img)
            self._ROC_curve(target_change_map, difference_img)

        self._compute_metrics(
            target_change_map, difference_img, self.difference_img_metrics
        )

        change_map = self._change_map(difference_img)
        self._compute_metrics(target_change_map, change_map, self.change_map_metrics)

        tf.print(
            "Kappa:",
            self.metrics_history["Kappa"][-1],
            "Accuracy:",
            self.metrics_history["ACC"][-1],
        )
        confusion_map = self._confusion_map(target_change_map, change_map)

        return confusion_map

    def final_evaluate(self, evaluation_dataset, save_images, final_filter, **kwargs):
        self.epoch.assign_add(1)
        tf.summary.experimental.set_step(self.epoch)
        self._save_images.assign(save_images)
        for eval_data in evaluation_dataset.batch(1):
            ev_res = self.evaluate(*eval_data, final_filter)
        self._save_images.assign(False)
        tf.summary.flush(self.tb_writer)
        self._write_metric_history()

    def _compute_metrics(self, y_true, y_pred, metrics):
        y_true, y_pred = tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1])
        for name, metric in metrics.items():
            metric.update_state(y_true, y_pred)
            self.metrics_history[name].append(metric.result().numpy())

            with tf.device("cpu:0"):
                with self.tb_writer.as_default():
                    tf.summary.scalar(name, metric.result())

            metric.reset_states()

    def _write_metric_history(self):
        for name, history in self.metrics_history.items():
            with open(self.log_path + "/" + name + ".txt", "w") as f:
                f.write(str(history))

    @image_to_tensorboard()
    def print_image(self, x):
        return x

    def print_all_input_images(self, evaluation_dataset):
        tf.summary.experimental.set_step(self.epoch + 1)
        self._save_images.assign(True)
        for x, y, z in evaluation_dataset.batch(1):
            self.print_image(x, name="x")
            self.print_image(y, name="y")
            self.print_image(tf.cast(z, dtype=tf.float32), name="Ground_Truth")
        self._save_images.assign(False)
        tf.summary.flush(self.tb_writer)

    def save_model(self):
        print("ChangeDetector.save_model() is not implemented")

    @image_to_tensorboard(static_name="z_ROC_Curve")
    def _ROC_curve(self, y_true, y_pred):
        y_true, y_pred = tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1])
        # print('cw',y_true,y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        fig = plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic curve")
        plt.legend(loc="lower right")
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        data = tf.convert_to_tensor(
            data.reshape(fig.canvas.get_width_height()[::-1] + (3,))[np.newaxis, ...],
            dtype=tf.float32,
        )
        plt.close()
        return data


def test(DATASET="Texas"):
    CONFIG = get_config(DATASET)
    _, _, EVALUATE, _ = datasets.fetch(DATASET, **CONFIG)
    cd = ChangeDetector(**CONFIG)
    cd.print_all_input_images(EVALUATE)


if __name__ == "__main__":
    test("Texas")
