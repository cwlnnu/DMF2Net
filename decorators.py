import tensorflow as tf
from functools import wraps
from timeit import default_timer as timer


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = timer()
        out = func(*args, **kwargs)
        stop = timer()

        return stop - start, out

    return wrapper


def _change_image_range(tensor):
    return (tensor - tf.reduce_min(tensor)) / (
        tf.reduce_max(tensor) - tf.reduce_min(tensor)
    )


def write_image_to_summary(image, writer, name, pre_process=None):
    if image.dtype == tf.bool:
        image = tf.cast(image, tf.float32)

    image = _change_image_range(image)
    if pre_process is not None:
        image = pre_process(image)

    with tf.device("cpu:0"):
        with writer.as_default():
            tf.summary.image(name, image)


def write_image_to_png(image, filename):
    if tf.rank(image) == 4:
        image = image[0]
    image = _change_image_range(image)
    image = tf.cast(255 * image, tf.uint8)
    contents = tf.image.encode_png(image)
    tf.io.write_file(filename, contents)


def image_to_tensorboard(static_name=None, pre_process=None):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, name=None, **kwargs):
            name = name if name is not None else static_name
            out = tmp = func(self, *args, **kwargs)
            if tmp.shape[-1] > 3:
                if "x" in name:
                    ch = self.channels["x"]
                    name += str(self.channels["x"])
                elif "y" in name:
                    ch = self.channels["y"]
                    name += str(self.channels["y"])
                tmp2 = tf.concat(
                    [
                        tf.expand_dims(tmp[..., ch[0]], -1),
                        tf.expand_dims(tmp[..., ch[1]], -1),
                        tf.expand_dims(tmp[..., ch[2]], -1),
                    ],
                    3,
                )
            else:
                tmp2 = tmp
            if (
                name is not None
                and self.evaluation_frequency > 0
                and not tf.cast(
                    tf.summary.experimental.get_step() % self.evaluation_frequency,
                    dtype=tf.bool,
                )
            ) or self._save_images:
                write_image_to_summary(tmp2, self.tb_writer, name, pre_process)
            if self._save_images and name is not None:
                filename = self._image_dir + tf.constant(f"/{name}.png")
                write_image_to_png(tmp2, filename)
            return out

        return wrapper

    return decorator
