from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    UpSampling2D,
    Dense,
    GlobalAveragePooling2D,
    Reshape,
)
from tensorflow.keras.activations import relu, sigmoid, tanh
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import math
import numpy as np

class conv1_kernel_initializer(tf.keras.initializers.Initializer):
    def __init__(self, kernel_diff):
        self.kernel_diff = kernel_diff
    def __call__(self, shape, dtype = None):
        kernel_init = tf.convert_to_tensor(self.kernel_diff)
        return kernel_init


class Conv2d_cd(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.8):

        super(Conv2d_cd, self).__init__()
        self.bias = bias
        self.out_channels = out_channels
        self.conv = Conv2D(out_channels, kernel_size=kernel_size, strides=stride, padding='same', dilation_rate=dilation, groups=groups, use_bias=bias)
        self.theta = theta

    def call(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            tmp = self.conv.get_weights()
            tmp1 = tmp[0]
            kernel_diff = tmp1.sum(0).sum(0)
            kernel_diff = kernel_diff[None, None, :, :]

            conv1 = Conv2D(self.out_channels, kernel_size=1, strides=1, padding='same', dilation_rate=1,
                                groups=1, use_bias=False,kernel_initializer=conv1_kernel_initializer(kernel_diff))

            out_diff = conv1(x)

            return out_normal - self.theta * out_diff


class iAFF(tf.keras.layers.Layer):
    def __init__(self, channels=50, r=4):
        super().__init__()
        inter_channels = int(channels//r)

        self.local_conv1_1 = Conv2D(inter_channels, kernel_size=1, strides=1, padding='same', dilation_rate=1)
        self.local_conv1_2 = Conv2D(channels, kernel_size=1, strides=1, padding='same', dilation_rate=1)
        self.local_conv2_1 = Conv2D(inter_channels, kernel_size=1, strides=1, padding='same', dilation_rate=1)
        self.local_conv2_2 = Conv2D(channels, kernel_size=1, strides=1, padding='same', dilation_rate=1)
        self.local_conv3_1 = Conv2D(inter_channels, kernel_size=1, strides=1, padding='same', dilation_rate=1)
        self.local_conv3_2 = Conv2D(channels, kernel_size=1, strides=1, padding='same', dilation_rate=1)

        self.avg1 = GlobalAveragePooling2D()
        self.global_conv1_1 = Conv2D(inter_channels, kernel_size=1, strides=1, padding='same', dilation_rate=1)
        self.global_conv1_2 = Conv2D(channels, kernel_size=3, strides=1, padding='same', dilation_rate=1)
        self.avg2 = GlobalAveragePooling2D()
        self.global_conv2_1 = Conv2D(inter_channels, kernel_size=1, strides=1, padding='same', dilation_rate=1)
        self.global_conv2_2 = Conv2D(channels, kernel_size=3, strides=1, padding='same', dilation_rate=1)
        self.avg3 = GlobalAveragePooling2D()
        self.global_conv3_1 = Conv2D(inter_channels, kernel_size=1, strides=1, padding='same', dilation_rate=1)
        self.global_conv3_2 = Conv2D(channels, kernel_size=3, strides=1, padding='same', dilation_rate=1)

    def call(self, x_level_0, x_level_1):
        xa = x_level_0 + x_level_1
        x_local = self.local_conv1_1(xa)
        x_local = relu(x_local, alpha=0.2)
        x_local = self.local_conv1_2(x_local)
        x_global = self.avg1(xa)
        x_global = Reshape((1, 1, x_global.shape[1]))(x_global)
        x_global = self.global_conv1_1(x_global)
        x_global = relu(x_global, alpha=0.2)
        x_global = self.global_conv1_2(x_global)
        xlg = x_local + x_global
        weight = tf.nn.sigmoid(xlg)
        out = x_level_0 * weight + x_level_1 * (1-weight)

        x_local1 = self.local_conv2_1(out)
        x_local1 = relu(x_local1, alpha=0.2)
        x_local1 = self.local_conv2_2(x_local1)
        x_global1 = self.avg2(out)
        x_global1 = Reshape((1, 1, x_global1.shape[1]))(x_global1)
        x_global1 = self.global_conv2_1(x_global1)
        x_global1 = relu(x_global1, alpha=0.2)
        x_global1 = self.global_conv2_2(x_global1)
        xlg1 = x_local1 + x_global1
        weight1 = tf.nn.sigmoid(xlg1)
        out1 = x_level_0 * weight1 + x_level_1 * (1 - weight1)

        return out1


class ImageTranslationNetwork(Model):

    def __init__(
        self,
        input_chs,
        filter_spec,
        name,
        l2_lambda=1e-3,
        leaky_alpha=0.3,
        dropout_rate=0.2,
        dtype="float32",
    ):
        super().__init__(name=name, dtype=dtype)

        self.leaky_alpha = leaky_alpha
        self.dropout = Dropout(dropout_rate, dtype=dtype)
        conv_specs = {
            "kernel_size": 3,
            "strides": 1,
            "kernel_initializer": "GlorotNormal",
            "padding": "same",
            "kernel_regularizer": l2(l2_lambda),
            "dtype": dtype,
        }

        self.layers_ = []
        for l, n_filters in enumerate(filter_spec):
            if l == 0:
                layer = Conv2D(
                    n_filters,
                    input_shape=(None, None, input_chs),
                    name=f"{name}-{l:02d}",
                    **conv_specs,
                )
            else:
                layer = Conv2D(n_filters, name=f"{name}-{l:02d}", **conv_specs)
            self.layers_.append(layer)
            if "enc" in name:
                if l < len(filter_spec) // 2:
                    self.layers_.append(MaxPooling2D(name=f"{name}-MP_{l:02d}", padding='same'))
                else:
                    if l < len(filter_spec) - 1:
                        self.layers_.append(UpSampling2D(name=f"{name}-UP_{l:02d}"))

        self.diff_conv2 = Conv2d_cd(50)
        self.iaff = iAFF()

    def call(self, x, training=False):
        skips = []
        for layer in self.layers_[:-1]:
            if "MP" in layer.name:
                diff2 = self.diff_conv2(x)
                diff2 = relu(diff2, alpha=self.leaky_alpha)
                skips.append(diff2)
                x = layer(x)
            elif "UP" in layer.name:
                x = layer(x)
                if x.shape[1] == 876:
                    x = x[:, :875, :, :]
                x = self.iaff(x, skips.pop())
            else:
                x = self.dropout(x, training)
                x = layer(x)
                x = relu(x, alpha=self.leaky_alpha)
        x = self.dropout(x, training)
        x = self.layers_[-1](x)
        x = tanh(x)
        return x
