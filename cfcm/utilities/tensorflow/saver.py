import os

import tensorflow as tf


class Saver(object):
    def __init__(self, sess, input_dict, output_dict, path):
        self.sess = sess
        self.input_dict = input_dict
        self.output_dict = output_dict
        self.path = path
        self.iteration = 0

        self.input_dict_info = {}
        for key in input_dict.keys():
            self.input_dict_info[key] = \
                tf.saved_model.utils.build_tensor_info(
                    self.input_dict[key]
                )

        self.output_dict_info = {}
        for key in output_dict.keys():
            self.output_dict_info[key] = \
                tf.saved_model.utils.build_tensor_info(
                    self.output_dict[key]
                )

        self.prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=self.input_dict_info,
                outputs=self.output_dict_info)
        )

    def save(self):
        self.iteration += 1

        export_path = os.path.join(
            tf.compat.as_bytes(self.path),
            tf.compat.as_bytes(str(self.iteration))
        )

        self.builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        self.builder.add_meta_graph_and_variables(
            self.sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'prediction': self.prediction_signature,
            }
        )

        self.builder.save()
