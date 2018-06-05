import numpy as np


class FitRunner(object):
    outputs = {}
    losses = {}
    placeholders = {}
    summaries = {}

    sess = None
    iteration = 0

    def __init__(self):
        self.create_placeholders()
        self.create_outputs()
        self.create_losses()
        self.create_summaries()
        self.create_training_op()

    def create_outputs(self):
        self.outputs = {}

    def create_losses(self):
        self.losses = {}

    def create_placeholders(self):
        self.placeholders = {}

    def create_summaries(self):
        self.summaries = {}

    def create_training_op(self):
        raise NotImplementedError

    def run_epoch_train(self):
        raise NotImplementedError

    def run_epoch_valid(self):
        raise NotImplementedError

    def run_iteration(self, feed_dict, op_list, summaries):
        output_args = self.sess.run(op_list, feed_dict=feed_dict)
        return output_args
