import numpy as np
import tensorflow as tf


class SummaryManager(object):
    def __init__(self, sess, summary_dir, summary_group_name='default'):
        self.sess = sess
        self.summary_data = {}
        self.summary_type = {}
        self.summary_placeholders = {}
        self.summary_ops = []

        self.writer = tf.summary.FileWriter(summary_dir + '/' + summary_group_name, self.sess.graph)

    def add_data_to_summary(self, summary_type, summary_name, summary_value):
        assert summary_type in ['image', 'scalar']

        if summary_type == 'scalar':
            try:
                self.summary_data[summary_name].append(summary_value)
            except KeyError:
                self.summary_placeholders[summary_name] = \
                    tf.placeholder(shape=(), name=summary_name, dtype=tf.float32)

                self.summary_data[summary_name] = [summary_value]

                self.summary_type[summary_name] = 'scalar'

                self.summary_ops.append(
                    tf.summary.scalar(summary_name, self.summary_placeholders[summary_name])
                )

        elif summary_type == 'image':
            try:
                # assert np.all(self.summary_placeholders[summary_name].get_shape().as_list()[1:4] == summary_value.shape)
                self.summary_data[summary_name].append(summary_value)
            except KeyError:
                assert len(summary_value.shape) == 3

                self.summary_placeholders[summary_name] = \
                    tf.placeholder(shape=[None] + list(summary_value.shape), name=summary_name, dtype=tf.float32)

                self.summary_data[summary_name] = [summary_value.astype(np.float32)]

                self.summary_type[summary_name] = 'image'

                self.summary_ops.append(
                    tf.summary.image(summary_name, self.summary_placeholders[summary_name])
                )

        # TODO histogram

    def reset(self):
        for key in self.summary_data:
            self.summary_data[key] = []

    def write(self, iteration, additional_summary_results=[]):
        feed_dict = {}
        for key in self.summary_data:
            if self.summary_type[key] == 'scalar':
                feed_dict[self.summary_placeholders[key]] = np.mean(self.summary_data[key])

            elif self.summary_type[key] == 'image':
                feed_dict[self.summary_placeholders[key]] = np.asarray(self.summary_data[key], dtype=np.float32)

        summaries = self.sess.run(self.summary_ops, feed_dict=feed_dict)

        summaries += additional_summary_results
        for summary in summaries:
            self.writer.add_summary(summary, iteration)

        self.writer.flush()

        self.reset()
