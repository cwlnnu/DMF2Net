import numpy as np
import tensorflow as tf
from decorators import write_image_to_png
from tqdm import trange
import re


def _border_trim_slice(n, ps, pstr):

    t_n = (n - ps) % pstr
    t_start, t_end = t_n // 2, t_n // 2 + t_n % 2
    assert t_start < n - t_end
    return slice(t_start, n - t_end)


def remove_borders(x, ps, pstr=None):
    if pstr == 1:
        return x
    if pstr == None:
        pstr = ps
    h, w = x.shape[:2]
    slice_h = _border_trim_slice(h, ps, pstr)
    slice_w = _border_trim_slice(w, ps, pstr)

    return x[slice_h, slice_w, ...]


def affinity(x):
    _, h, w, c = x.shape
    x_1 = tf.expand_dims(tf.reshape(x, [-1, h * w, c]), 2)
    x_2 = tf.expand_dims(tf.reshape(x, [-1, h * w, c]), 1)
    A = tf.norm(x_1 - x_2, axis=-1)
    krnl_width = tf.math.top_k(A, k=A.shape[-1]).values
    krnl_width = tf.reduce_mean(input_tensor=krnl_width[:, :, (h * w) // 4], axis=1)
    krnl_width = tf.reshape(krnl_width, (-1, 1, 1))
    krnl_width = tf.where(
        tf.math.equal(krnl_width, tf.zeros_like(krnl_width)),
        tf.ones_like(krnl_width),
        krnl_width,
    )
    A = tf.exp(-(tf.divide(A, krnl_width) ** 2))
    return A


def Degree_matrix(x, y):
    ax = affinity(x)
    ay = affinity(y)
    D = tf.norm(tf.expand_dims(ax, 1) - tf.expand_dims(ay, 2), 2, -1)
    min_D = tf.reduce_min(D, axis=[1, 2], keepdims=True)
    max_D = tf.reduce_max(D, axis=[1, 2], keepdims=True)
    D = (D - min_D) / (max_D - min_D)
    return D


def ztz(x, y):
    flat_shape = [x.shape[0], x.shape[1] ** 2, -1]
    x = tf.reshape(x, flat_shape)
    y = tf.reshape(y, flat_shape)
    norms = tf.norm(tf.concat([x, y], 1), axis=-1, keepdims=True)
    max_norm = tf.math.reduce_max(norms, axis=1, keepdims=True)

    ztz = (tf.keras.backend.batch_dot(y, x, -1) + max_norm) / (2 * max_norm)

    return ztz


if __name__ == "__main__":
    pass
