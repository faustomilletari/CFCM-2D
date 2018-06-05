from __future__ import division

import tensorflow as tf
from tensorflow.contrib.layers import (
    conv2d,
    conv2d_transpose,
    max_pool2d,
    batch_norm
)

from .tf_resnet import seg_resnet_v2, seg_resnet_lstm_v2


def residual_block(net, is_training, n_filters):
    net = batch_norm(net, is_training=is_training, activation_fn=tf.nn.relu)
    net = conv2d(net, n_filters, [3, 3], activation_fn=None)
    net = batch_norm(net, is_training=is_training, activation_fn=tf.nn.relu)
    net = conv2d(net, n_filters, [3, 3], activation_fn=None)

    return net


def upsample_tf(factor, input_img):
    input_image_shape = input_img.get_shape().as_list()
    new_height = int(round(input_image_shape[1] * factor))
    new_width = int(round(input_image_shape[2] * factor))
    resized = tf.image.resize_images(input_img, [new_height, new_width])

    return resized


def lstm_resnet(input, is_training, style, num_output_channels):
    model = seg_resnet_lstm_v2(resnet_size=style, num_classes=num_output_channels, data_format='channels_last')
    output = model(inputs=input, is_training=is_training)

    return output


def resnet(input, is_training, style, num_output_channels):
    model = seg_resnet_v2(resnet_size=style, num_classes=num_output_channels, data_format='channels_last')
    output = model(inputs=input, is_training=is_training)

    return output
