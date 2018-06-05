from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import slim

from cfcm.layers.layers import BasicConvLSTMCell

BASE_NUM_KERNELS = 64


def batch_norm_relu(inputs, is_training):
    net = slim.batch_norm(inputs, is_training=is_training)
    net = tf.nn.relu(net)
    return net


def conv2d_transpose(inputs, output_channels, kernel_size):
    return tf.contrib.slim.conv2d_transpose(
        inputs,
        num_outputs=output_channels,
        kernel_size=kernel_size,
        stride=2,
    )


def conv2d_fixed_padding(inputs, filters, kernel_size, stride):
    net = slim.conv2d(inputs,
                      filters,
                      kernel_size,
                      stride=stride,
                      padding=('SAME' if stride == 1 else 'VALID'),
                      activation_fn=None
                      )
    return net


def building_block(inputs, filters, is_training, projection_shortcut, stride):
    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training)
    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, stride=stride)

    inputs = batch_norm_relu(inputs, is_training)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, stride=1)

    return inputs + shortcut


def bottleneck_block(inputs, filters, is_training, projection_shortcut, stride):
    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, stride=1)

    inputs = batch_norm_relu(inputs, is_training)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, stride=stride)

    inputs = batch_norm_relu(inputs, is_training)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=4 * filters, kernel_size=1, stride=1)

    return inputs + shortcut


def block_layer_compressing(inputs, filters, block_fn, blocks, stride, is_training, name):
    filters_out = 4 * filters if block_fn is bottleneck_block else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1, stride=stride)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, is_training, projection_shortcut, stride)

    layers_outputs = [inputs]

    for i in range(1, blocks):
        inputs = block_fn(inputs, filters, is_training, None, 1)

        layers_outputs.append(tf.nn.relu(inputs))

    return tf.identity(inputs, name), layers_outputs


def block_layer_expanding(inputs, forwarded_feature_list, filters, block_fn, blocks, stride, is_training, name,
                          concat_zero=True):
    filters_out = 4 * filters if block_fn is bottleneck_block else filters

    if concat_zero:
        inputs = tf.concat([inputs, forwarded_feature_list[0]], axis=-1)

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1, stride=stride)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, is_training, projection_shortcut, stride)

    layers_outputs = [inputs]

    for i in range(1, blocks):
        inputs = tf.concat([inputs, forwarded_feature_list[i]], axis=-1)
        inputs = block_fn(inputs, filters, is_training, projection_shortcut, 1)

        layers_outputs.append(tf.nn.relu(inputs))

    return tf.identity(inputs, name), layers_outputs


def seg_resnet_lstm_v2_generator(block_fn, layers, num_classes, data_format='channels_last'):
    def model(inputs, is_training):
        if block_fn is bottleneck_block:
            factor = 4
            base_num_kernels = int(BASE_NUM_KERNELS / 4)
        else:
            factor = 1
            base_num_kernels = BASE_NUM_KERNELS

        inputs = conv2d_fixed_padding(inputs=inputs, filters=base_num_kernels * factor, kernel_size=7, stride=1)

        inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=2, strides=2, padding='SAME')

        output_b1, output_list_b1 = block_layer_compressing(
            inputs=inputs,
            filters=base_num_kernels,
            block_fn=block_fn,
            blocks=layers[0],
            stride=1,
            is_training=is_training,
            name='block_layer1'
        )

        output_b1 = tf.layers.max_pooling2d(
            inputs=output_b1, pool_size=2, strides=2, padding='SAME',
            data_format=data_format
        )

        output_b2, output_list_b2 = block_layer_compressing(
            inputs=output_b1, filters=base_num_kernels * 2, block_fn=block_fn, blocks=layers[1],
            stride=1, is_training=is_training, name='block_layer2'
        )

        output_b2 = tf.layers.max_pooling2d(
            inputs=output_b2, pool_size=2, strides=2, padding='SAME',
            data_format=data_format
        )

        output_b3, output_list_b3 = block_layer_compressing(
            inputs=output_b2, filters=base_num_kernels * 4, block_fn=block_fn, blocks=layers[2],
            stride=1, is_training=is_training, name='block_layer3'
        )

        output_b3 = tf.layers.max_pooling2d(
            inputs=output_b3, pool_size=2, strides=2, padding='SAME',
            data_format=data_format
        )

        output_b4, output_list_b4 = block_layer_compressing(
            inputs=output_b3, filters=base_num_kernels * 8, block_fn=block_fn, blocks=layers[3],
            stride=1, is_training=is_training, name='block_layer4'
        )

        # lstm - segmentation path

        initial_hidden = tf.zeros_like(output_b4)
        initial_cell = tf.zeros_like(output_b4)

        initial_state = tf.concat([initial_cell, initial_hidden], axis=3)

        shape = [output_b4.get_shape().as_list()[1], output_b4.get_shape().as_list()[2]]
        lstm_b4 = BasicConvLSTMCell(shape, [3, 3], num_features=base_num_kernels * 8 * factor, scope='lstm_b4',
                                    activation=tf.nn.relu)

        _, state = rnn.static_rnn(lstm_b4, output_list_b4, initial_state=initial_state, dtype=tf.float32)

        state = conv2d_transpose(state, kernel_size=2, output_channels=base_num_kernels * 8 * factor)

        shape = [output_b3.get_shape().as_list()[1], output_b3.get_shape().as_list()[2]]
        lstm_b3 = BasicConvLSTMCell(shape, [3, 3], num_features=base_num_kernels * 4 * factor, scope='lstm_b3',
                                    activation=tf.nn.relu)

        _, state = rnn.static_rnn(lstm_b3, output_list_b3, initial_state=state, dtype=tf.float32)

        state = conv2d_transpose(state, kernel_size=2, output_channels=base_num_kernels * 4 * factor)

        shape = [output_b2.get_shape().as_list()[1], output_b2.get_shape().as_list()[2]]
        lstm_b2 = BasicConvLSTMCell(shape, [3, 3], num_features=base_num_kernels * 2 * factor, scope='lstm_b2',
                                    activation=tf.nn.relu)

        _, state = rnn.static_rnn(lstm_b2, output_list_b2, initial_state=state, dtype=tf.float32)

        state = conv2d_transpose(state, kernel_size=2, output_channels=base_num_kernels * 2 * factor)

        shape = [output_b1.get_shape().as_list()[1], output_b1.get_shape().as_list()[2]]
        lstm_b1 = BasicConvLSTMCell(shape, [3, 3], num_features=base_num_kernels * factor, scope='lstm_b1',
                                    activation=tf.nn.relu)

        output, state = rnn.static_rnn(lstm_b1, output_list_b1, initial_state=state, dtype=tf.float32)

        hidden = conv2d_transpose(output[-1], kernel_size=2, output_channels=base_num_kernels * factor)

        conv_final = conv2d_fixed_padding(inputs=hidden, filters=base_num_kernels * factor, kernel_size=3, stride=1)

        outputs = conv2d_fixed_padding(inputs=conv_final, filters=num_classes, kernel_size=3, stride=1)

        return outputs

    return model


def seg_resnet_lstm_v2(resnet_size, num_classes, data_format=None):
    model_params = {
        18: {'block': building_block, 'layers': [2, 2, 2, 2]},
        34: {'block': building_block, 'layers': [3, 4, 6, 3]},
        68: {'block': building_block, 'layers': [3, 4, 23, 3]},
        50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
        101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
        152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
        200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
    }

    if resnet_size not in model_params:
        raise ValueError('Not a valid resnet_size:', resnet_size)

    params = model_params[resnet_size]
    return seg_resnet_lstm_v2_generator(params['block'], params['layers'], num_classes, data_format)


def seg_resnet_v2_generator(block_fn, layers, num_classes, data_format='channels_last'):
    def model(inputs, is_training):
        if block_fn is bottleneck_block:
            factor = 4
            base_num_kernels = int(BASE_NUM_KERNELS / 4)
        else:
            factor = 1
            base_num_kernels = BASE_NUM_KERNELS

        inputs = conv2d_fixed_padding(inputs=inputs, filters=base_num_kernels * factor, kernel_size=7, stride=1)

        inputs = tf.layers.max_pooling2d(
            inputs=inputs, pool_size=2, strides=2, padding='SAME',
            data_format=data_format)

        # compressing path

        output_b1, output_list_b1 = block_layer_compressing(
            inputs=inputs, filters=base_num_kernels, block_fn=block_fn, blocks=layers[0],
            stride=1, is_training=is_training, name='block_layer1_1'
        )

        output_b1 = tf.layers.max_pooling2d(
            inputs=output_b1, pool_size=2, strides=2, padding='SAME',
            data_format=data_format
        )

        output_b2, output_list_b2 = block_layer_compressing(
            inputs=output_b1, filters=base_num_kernels * 2, block_fn=block_fn, blocks=layers[1],
            stride=1, is_training=is_training, name='block_layer2_1'
        )

        output_b2 = tf.layers.max_pooling2d(
            inputs=output_b2, pool_size=2, strides=2, padding='SAME',
            data_format=data_format)

        output_b3, output_list_b3 = block_layer_compressing(
            inputs=output_b2, filters=base_num_kernels * 4, block_fn=block_fn, blocks=layers[2],
            stride=1, is_training=is_training, name='block_layer3_1'
        )

        output_b3 = tf.layers.max_pooling2d(
            inputs=output_b3, pool_size=2, strides=2, padding='SAME',
            data_format=data_format)

        output_b4, output_list_b4 = block_layer_compressing(
            inputs=output_b3, filters=base_num_kernels * 8, block_fn=block_fn, blocks=layers[3],
            stride=1, is_training=is_training, name='block_layer4_1'

        )

        features = output_list_b4[-1]
        output_list_b4 = output_list_b4[::-1]

        # expanding path

        features, _ = block_layer_expanding(
            inputs=features, forwarded_feature_list=output_list_b4, filters=base_num_kernels * 8 * factor, block_fn=block_fn,
            blocks=len(output_list_b4), stride=1, is_training=is_training, name='block_layer4_2', concat_zero=False
        )  # here concat_zero is false because we have nothing to concat from prev blocks of decompressing path...

        features = conv2d_transpose(features, kernel_size=2, output_channels=base_num_kernels * 4 * factor)

        features, _ = block_layer_expanding(
            inputs=features, forwarded_feature_list=output_list_b3[::-1], filters=base_num_kernels * 4 * factor,
            block_fn=block_fn,
            blocks=len(output_list_b3[::-1]), stride=1, is_training=is_training, name='block_layer3_2'

        )

        features = conv2d_transpose(features, kernel_size=2, output_channels=base_num_kernels * 2 * factor)

        features, _ = block_layer_expanding(
            inputs=features, forwarded_feature_list=output_list_b2[::-1], filters=base_num_kernels * 2 * factor,
            block_fn=block_fn,
            blocks=len(output_list_b2[::-1]), stride=1, is_training=is_training, name='block_layer2_2'

        )

        features = conv2d_transpose(features, kernel_size=2, output_channels=base_num_kernels * factor)

        features, _ = block_layer_expanding(
            inputs=features, forwarded_feature_list=output_list_b1[::-1], filters=base_num_kernels * factor,
            block_fn=block_fn,
            blocks=len(output_list_b1[::-1]), stride=1, is_training=is_training, name='block_layer1_2'

        )
        features = conv2d_transpose(features, kernel_size=2, output_channels=base_num_kernels * factor)

        conv_final = conv2d_fixed_padding(inputs=features, filters=base_num_kernels * factor, kernel_size=3, stride=1)

        outputs = conv2d_fixed_padding(inputs=conv_final, filters=num_classes, kernel_size=3, stride=1)

        return outputs

    return model


def seg_resnet_v2(resnet_size, num_classes, data_format=None):
    model_params = {
        18: {'block': building_block, 'layers': [2, 2, 2, 2]},
        34: {'block': building_block, 'layers': [3, 4, 6, 3]},
        68: {'block': building_block, 'layers': [3, 4, 23, 3]},
        50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
        101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
        152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
        200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
    }

    if resnet_size not in model_params:
        raise ValueError('Not a valid resnet_size:', resnet_size)

    params = model_params[resnet_size]
    return seg_resnet_v2_generator(params['block'], params['layers'], num_classes, data_format)
