import tensorflow as tf

EPS = 0.0000001


def dice(labels, prediction):
    with tf.variable_scope('dice'):
        dc = 2.0 * \
             tf.reduce_sum(labels * prediction, axis=[1, 2]) / \
             tf.reduce_sum(labels ** 2 + prediction ** 2, axis=[1, 2]) + EPS

    return dc


def dice_loss(labels, prediction):
    with tf.variable_scope('dice_loss'):
        dl = 1.0 - dice(labels, prediction)

    return dl


def binary_cross_entropy_2D(labels, logits, reweight=False):
    labels_shape = labels.get_shape().as_list()

    pixel_size = labels_shape[1] * labels_shape[2]

    logits = tf.reshape(logits, [-1, pixel_size])

    labels = tf.reshape(labels, [-1, pixel_size])

    number_foreground = tf.reduce_sum(labels)
    number_background = tf.reduce_sum(1.0 - labels)

    weight_foreground = number_background / (number_foreground + EPS)
    if reweight:
        loss = \
            tf.nn.weighted_cross_entropy_with_logits(
                targets=tf.cast(labels, tf.float32),
                logits=logits,
                pos_weight=weight_foreground
            )

    else:
        loss = \
            tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.cast(labels, tf.float32),
                logits=logits,
            )

    loss = tf.reduce_mean(loss)

    return loss
