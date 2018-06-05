import os
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

import numpy as np
import tensorflow as tf

from cfcm.data.managers import DataManager
from cfcm.data.loading import load_EndoVis_transform, load_montgomery_xray
from cfcm.data.transforms import (
    add_trailing_singleton_dimension_transform,
    threshold_labels_transform,
    normalize_between_zero_and_one,
    resize_img,
    flip_left_right,
    flip_up_down,
    specularity,
    crop_actual_image,
    make_onehot,
    convert_to_float,
    shuffle_data,
)
from cfcm.layers.losses import dice, dice_loss, binary_cross_entropy_2D
from cfcm.utilities.fitting.fitrunners import FitRunner
from cfcm.utilities.iterators.batchiterators import BatchIterator
from cfcm.utilities.tensorflow.saver import Saver
from cfcm.utilities.tensorflow.summary import SummaryManager
from cfcm.utilities.timing import timeit
from cfcm.utilities.visualization import surfd
from .models import lstm_resnet, resnet


class MontgomeryXRay(object):
    dataset_name = 'XRAY'

    data_size = [256, 256]
    data_channels = 1
    num_output_channels = 1

    number_deformation_control_points = 5
    max_displacement_std = 10

    batch_size = 32

    active_loss = None

    def __init__(self, jdata, data_folder):
        self.graph = tf.Graph()
        self.sess = tf.Session

        super(MontgomeryXRay, self).__init__()

        self.train_set_images = MontgomeryXRay._concatenate_data_path(data_folder, jdata['dataset']['training_images'])

        self.train_set_labels_l = MontgomeryXRay._concatenate_data_path(data_folder, jdata['dataset']['training_labels_l'])

        self.train_set_labels_r = MontgomeryXRay._concatenate_data_path(data_folder, jdata['dataset']['training_labels_r'])

        self.valid_set_images = MontgomeryXRay._concatenate_data_path(data_folder, jdata['dataset']['testing_images'])

        self.valid_set_labels_l = MontgomeryXRay._concatenate_data_path(data_folder, jdata['dataset']['testing_labels_l'])

        self.valid_set_labels_r = MontgomeryXRay._concatenate_data_path(data_folder, jdata['dataset']['testing_labels_r'])

        # Data loading recipe
        self.training_data = DataManager([
            load_montgomery_xray(self.train_set_images, self.train_set_labels_l, self.train_set_labels_r),
            crop_actual_image(),
            resize_img(self.data_size[0]),

        ]
        )

        self.validation_data = DataManager([
            load_montgomery_xray(self.valid_set_images, self.valid_set_labels_l, self.valid_set_labels_r),
            crop_actual_image(),
            resize_img(self.data_size[0])
        ]
        )

        # Batch iterator recipe
        self.batch_iterator_train = BatchIterator(
            epoch_transforms=[
                shuffle_data(['images', 'labels']),
            ],
            iteration_transforms=[
                # elastic_transform(self.number_deformation_control_points, self.max_displacement_std),
                add_trailing_singleton_dimension_transform(field='images'),
                add_trailing_singleton_dimension_transform(field='labels'),
                threshold_labels_transform(),
            ],
            batch_size=self.batch_size,
            iteration_keys=['images', 'labels']
        )

        self.batch_iterator_train(self.training_data)

        self.batch_iterator_valid = BatchIterator(
            epoch_transforms=[
            ],
            iteration_transforms=[
                add_trailing_singleton_dimension_transform(field='images'),
                add_trailing_singleton_dimension_transform(field='labels'),
                threshold_labels_transform(),
            ],
            batch_size=self.batch_size,
            iteration_keys=['images', 'labels']
        )

        self.batch_iterator_valid(self.validation_data)

        self.sess = tf.Session()

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.saver = Saver(
            self.sess,
            self.placeholders,
            self.outputs,
            self.savepath
        )

        self.summary_manager_train = SummaryManager(
            self.sess,
            self.summary_path,
            'train'
        )

        self.summary_manager_valid = SummaryManager(
            self.sess,
            self.summary_path,
            'valid'
        )

        self.csv_file = open(os.path.join(self.summary_path, 'csv_results.csv'), 'w')
        self.csv_file.write('{},{},{},{},{},{},{},{},{}{}'.format(
            'Iteration', 'Dice', 'DiceSTD', 'MAD', 'MAD_STD', 'RMS', 'RMS_STD', 'HD', 'HD_STD', '\n'
        ))

    @staticmethod
    def _concatenate_data_path(data_folder, path_list):
        return [os.path.join(data_folder, p) for p in path_list]

    def add_summaries(self, data, manager):
        data['sigmoid'] = np.concatenate(data['sigmoid'], axis=0)
        data['loss'] = np.mean(data['loss'])
        data['images'] = np.concatenate(data['images'], axis=0)
        data['labels'] = np.concatenate(data['labels'], axis=0)
        data['dice'] = np.concatenate(data['dice'], axis=0)

        sigmoid = data['sigmoid'][0, :, :, 0:1]
        segmentation = sigmoid > 0.5

        groundtruth = data['labels'][0, :, :, 0:1]

        manager.add_data_to_summary('image', 'sigmoid', sigmoid)
        manager.add_data_to_summary('image', 'segmentation', segmentation)
        manager.add_data_to_summary('image', 'original', data['images'][0, :, :, :])
        manager.add_data_to_summary('image', 'ground_truth', groundtruth)
        manager.add_data_to_summary('scalar', self.active_loss, data['loss'])
        manager.add_data_to_summary('scalar', 'dice', np.mean(data['dice']))

        manager.write(self.iteration)

    def add_validation_summaries(self, data):
        self.add_summaries(data, self.summary_manager_valid)

        iteration = self.iteration

        dice_list = data['dice']

        label_list = data['labels']
        seg_list = (data['sigmoid'] > 0.5).astype(np.float32)

        dice = np.mean(dice_list)
        dice_std = np.std(dice_list)

        msds = []
        rmss = []
        hds = []

        for gt, seg in zip(label_list, seg_list):
            surface_distance = surfd(gt, seg)

            msds.append(surface_distance.mean())
            rmss.append(np.sqrt((surface_distance ** 2).mean()))
            hds.append(surface_distance.max())

        self.csv_file.write('{},{},{},{},{},{},{},{},{}{}'.format(
            iteration, dice, dice_std, np.mean(msds), np.std(msds), np.mean(rmss), np.std(rmss), np.mean(hds),
            np.std(hds), '\n'
        ))

    def add_training_summaries(self, data):
        self.add_summaries(data, self.summary_manager_train)


class ENDOVIS2015(object):
    dataset_name = 'ENDOVIS15'

    datapath_train = 'Segmentation/Training'
    datapath_test = 'Segmentation/Testing'

    data_size = [256, 256]
    batch_size = 32

    data_channels = 3
    num_output_channels = 3

    data_channels = data_channels

    def __init__(self, jdata, data_folder):
        self.training_set = jdata['dataset']['training']
        self.validation_set = jdata['dataset']['testing']

        # Create list of paths for easy loading
        self.train_videos = \
            [os.path.join(data_folder, self.datapath_train, 'Dataset{}'.format(s), 'Video.avi') for s in self.training_set]
        self.train_videos_labels = \
            [os.path.join(data_folder, self.datapath_train, 'Dataset{}'.format(s), 'Segmentation.avi') for s in self.training_set]

        self.valid_videos = \
            [os.path.join(data_folder, self.datapath_test, 'Dataset{}'.format(s), 'Video.avi') for s in self.validation_set]
        self.valid_videos_labels = \
            [os.path.join(data_folder, self.datapath_test, 'Dataset{}'.format(s), 'Segmentation.avi') for s in self.validation_set]

        self.graph = tf.Graph()
        self.sess = tf.Session()

        super(ENDOVIS2015, self).__init__()

        transform_chain_train = \
            [
                load_EndoVis_transform(self.train_videos, self.train_videos_labels),
                resize_img(self.data_size[0]),
                convert_to_float(),
                threshold_labels_transform(),  # remove for multiclass!
                normalize_between_zero_and_one(),
            ]

        if self.num_output_channels > 1:
            transform_chain_train = \
                [
                    load_EndoVis_transform(self.train_videos, self.train_videos_labels),
                    resize_img(self.data_size[0]),
                    convert_to_float(),
                    make_onehot(self.num_output_channels),
                    normalize_between_zero_and_one(),
                ]

        transform_chain_valid = \
            [
                load_EndoVis_transform(self.valid_videos, self.valid_videos_labels),
                resize_img(self.data_size[0]),
                convert_to_float(),
                threshold_labels_transform(),  # remove for multiclass!
                normalize_between_zero_and_one(),
            ]

        if self.num_output_channels > 1:
            transform_chain_valid = \
                [
                    load_EndoVis_transform(self.valid_videos, self.valid_videos_labels),
                    resize_img(self.data_size[0]),
                    convert_to_float(),
                    make_onehot(self.num_output_channels),
                    normalize_between_zero_and_one(),
                ]

        # Data loading recipe
        self.training_data = DataManager(transform_chain_train)

        self.validation_data = DataManager(transform_chain_valid)

        assert self.validation_data['images'].shape[0] == self.validation_data['labels'].shape[0]

        # Batch iterator recipe
        self.batch_iterator_train = BatchIterator(
            epoch_transforms=[
                shuffle_data(['images', 'labels'])
            ],
            iteration_transforms=[
                flip_left_right(),
                flip_up_down(),
                specularity(),
                # add_trailing_singleton_dimension_transform('labels')
            ],
            batch_size=self.batch_size,
            iteration_keys=['images', 'labels']
        )

        self.batch_iterator_train(self.training_data)

        self.batch_iterator_valid = BatchIterator(
            epoch_transforms=[
                shuffle_data(['images', 'labels'])
            ],
            iteration_transforms=[
                # convert_to_float_and_correct_range(),
                # add_trailing_singleton_dimension_transform('labels')
            ],
            batch_size=self.batch_size,
            iteration_keys=['images', 'labels']
        )

        self.batch_iterator_valid(self.validation_data)

        self.sess = tf.Session()

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.saver = Saver(
            self.sess,
            self.placeholders,
            self.outputs,
            self.savepath
        )

        self.summary_manager_train = SummaryManager(
            self.sess,
            self.summary_path,
            'train'
        )

        self.summary_manager_valid = SummaryManager(
            self.sess,
            self.summary_path,
            'valid'
        )

        self.csv_file = open(os.path.join(self.summary_path, 'csv_results.csv'), 'w')
        self.csv_file.write('{},{},{},{},{},{},{},{},{},{},{},{}{}'.format(
            'Iteration', 'Dice', 'DiceSTD', 'MAD', 'MAD_STD', 'RMS', 'RMS_STD', 'HD', 'HD_STD', 'Accuracy', 'Recall',
            'Specificity', '\n'
        ))

    def add_summaries(self, data, manager):
        data['sigmoid'] = np.concatenate(data['sigmoid'], axis=0)
        data['loss'] = np.mean(data['loss'])
        data['images'] = np.concatenate(data['images'], axis=0)
        data['labels'] = np.concatenate(data['labels'], axis=0)
        data['dice'] = np.concatenate(data['dice'])

        sigmoid = data['sigmoid'][0, :, :, :]
        segmentation = sigmoid > 0.5

        groundtruth = data['labels'][0, :, :, :]

        manager.add_data_to_summary('image', 'sigmoid', sigmoid)
        manager.add_data_to_summary('image', 'segmentation', segmentation)
        manager.add_data_to_summary('image', 'original', data['images'][0, :, :, :])
        manager.add_data_to_summary('image', 'ground_truth', groundtruth)
        manager.add_data_to_summary('scalar', self.active_loss, data['loss'])
        manager.add_data_to_summary('scalar', 'dice', np.mean(data['dice']))

        manager.write(self.iteration)

    def add_validation_summaries(self, data):
        self.add_summaries(data, self.summary_manager_valid)
        iteration = self.iteration

        dice_list = data['dice']

        label_list = data['labels']
        seg_list = (data['sigmoid'] > 0.5).astype(np.float32)

        dice = np.mean(dice_list, axis=0)
        dice_std = np.std(dice_list, axis=0)

        for i in range(self.num_output_channels):
            recalls = np.zeros(len(label_list), dtype=np.float32)
            accuracies = np.zeros(len(label_list), dtype=np.float32)
            secificities = np.zeros(len(label_list), dtype=np.float32)

            print(len(label_list))
            for j, gt, seg in zip(range(len(label_list)), label_list, seg_list):
                tn, fp, fn, tp = confusion_matrix(gt[:, :, i].flatten(), seg[:, :, i].flatten()).ravel()
                specificity = float(tn) / float(tn + fp)

                accuracy = float(tp + tn) / float(tp + tn + fp + fn)
                recall = float(tp) / float(tp + fn)

                recalls[j] = recall
                accuracies[j] = accuracy
                secificities[j] = specificity

            print('{},{},{},{},{},{},{},{},{},{},{},{}{}'.format(
                iteration, dice[i], dice_std[i], 0, 0, 0, 0, 0, 0, np.mean(accuracies), np.mean(recalls),
                np.mean(secificities), '\n'
            ))

            self.csv_file.write('{},{},{},{},{},{},{},{},{},{},{},{}{}'.format(
                iteration, dice[i], dice_std[i], 0, 0, 0, 0, 0, 0, np.mean(accuracies), np.mean(recalls),
                np.mean(secificities), '\n'
            ))

    def add_training_summaries(self, data):
        self.add_summaries(data, self.summary_manager_train)


class SegBaseclass(FitRunner):
    active_loss = None

    def initialize_random_generators(self):
        np.random.seed(self.random_seed)
        tf.set_random_seed(self.random_seed)

    def create_placeholders(self):
        self.placeholders['images'] = tf.placeholder(
            shape=[None] + self.data_size + [self.data_channels],
            name='images',
            dtype=tf.float32
        )
        self.placeholders['is_training'] = tf.placeholder(
            shape=[],
            name='is_training',
            dtype=tf.bool
        )
        self.placeholders['labels'] = tf.placeholder(
            shape=[None] + self.data_size + [self.num_output_channels],
            name='labels',
            dtype=tf.float32
        )

    def create_outputs(self):
        raise NotImplementedError('Create_outputs is not defined in this class')

    def create_losses(self):
        if self.active_loss == 'reweighted_cross_entropy':
            self.losses['reweighted_cross_entropy'] = binary_cross_entropy_2D(
                self.placeholders['labels'], self.outputs['logits'], reweight=True
            )
        if self.active_loss == 'dice_loss':
            self.losses['dice_loss'] = tf.reduce_mean(
                dice_loss(self.placeholders['labels'], tf.cast(self.outputs['sigmoid'], tf.float32))
            )

    def create_training_op(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.fitting_op = self.optimizer.minimize(self.losses['dice'])

    def create_summaries(self):
        self.summaries = {'all': tf.summary.merge_all()}

    @timeit
    def run_epoch_train(self):
        raise NotImplementedError('The training loop is not defined in this class')

    @timeit
    def run_epoch_valid(self):
        raise NotImplementedError('The validation loop is not defined in this class')


class LSTMResNet(SegBaseclass):
    learning_rate = 0.00001
    batch_size = 16

    def __init__(self, jdata, model_folder):
        self.resnet_type = jdata['algorithm']['resnet_type']
        self.active_loss = jdata['algorithm']['loss']

        run_name = jdata['algorithm']['run_name']

        self.savepath = os.path.join(model_folder, self.dataset_name + '_LSTMResNet_' + run_name + self.active_loss)
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        self.summary_path = self.savepath + '/summaries'
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)

        super(LSTMResNet, self).__init__()

    def create_placeholders(self):
        self.placeholders['images'] = tf.placeholder(
            shape=[None] + self.data_size + [self.data_channels],
            name='images',
            dtype=tf.float32
        )
        self.placeholders['is_training'] = tf.placeholder(
            shape=[],
            name='is_training',
            dtype=tf.bool
        )
        self.placeholders['labels'] = tf.placeholder(
            shape=[None] + self.data_size + [self.num_output_channels],
            name='labels',
            dtype=tf.float32
        )

    def create_training_op(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(extra_update_ops):
            self.fitting_op = self.optimizer.minimize(self.losses[self.active_loss])

    def create_outputs(self):
        lstm_resnet_output = lstm_resnet(
            self.placeholders['images'], self.placeholders['is_training'], self.resnet_type, self.num_output_channels
        )

        self.outputs['logits'] = lstm_resnet_output

        if self.num_output_channels == 1:
            self.outputs['sigmoid'] = tf.nn.sigmoid(lstm_resnet_output)
            self.outputs['segmentation'] = self.outputs['sigmoid'] > 0.5
        else:
            self.outputs['sigmoid'] = tf.nn.softmax(lstm_resnet_output)
            self.outputs['segmentation'] = self.outputs['sigmoid'] > 0.5

        # not a dice loss, just dice coefficient
        self.outputs['dice'] = dice(self.placeholders['labels'], tf.cast(self.outputs['segmentation'], tf.float32))

    @timeit
    def run_epoch_train(self):
        self.iteration += 1

        summary_data = {
            'sigmoid': [],
            'loss': [],
            'images': [],
            'labels': [],
            'dice': [],
        }

        for batch in self.batch_iterator_train:
            _, loss, sigmoid, dice = self.run_iteration(
                feed_dict={
                    self.placeholders['images']: batch['images'],
                    self.placeholders['labels']: batch['labels'],
                    self.placeholders['is_training']: True,
                },
                op_list=[
                    self.fitting_op,
                    self.losses[self.active_loss],
                    self.outputs['sigmoid'],
                    self.outputs['dice'],
                ],
                summaries=[],
            )

            summary_data['sigmoid'].append(sigmoid)
            summary_data['loss'].append(loss)
            summary_data['images'].append(batch['images'])
            summary_data['labels'].append(batch['labels'])
            summary_data['dice'].append(dice)

        self.add_training_summaries(summary_data)

        if (self.iteration % 100) == 0:
            self.saver.save()

    @timeit
    def run_epoch_valid(self):
        summary_data = {
            'sigmoid': [],
            'loss': [],
            'images': [],
            'labels': [],
            'dice': [],
        }

        for batch in self.batch_iterator_valid:
            loss, sigmoid, dice = self.run_iteration(
                feed_dict={
                    self.placeholders['images']: batch['images'],
                    self.placeholders['labels']: batch['labels'],
                    self.placeholders['is_training']: True,
                },
                op_list=[
                    self.losses[self.active_loss],
                    self.outputs['sigmoid'],
                    self.outputs['dice'],
                ],
                summaries=[]
            )

            summary_data['sigmoid'].append(sigmoid)
            summary_data['loss'].append(loss)
            summary_data['images'].append(batch['images'])
            summary_data['labels'].append(batch['labels'])
            summary_data['dice'].append(dice)

        self.add_validation_summaries(summary_data)


class ResNet(LSTMResNet):
    learning_rate = 0.00001
    batch_size = 16

    def __init__(self, jdata, model_folder):
        self.resnet_type = jdata['algorithm']['resnet_type']
        self.active_loss = jdata['algorithm']['loss']

        run_name = jdata['algorithm']['run_name']
        self.savepath = os.path.join(model_folder, self.dataset_name + '_ResNet_' + run_name + self.active_loss)
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        self.summary_path = self.savepath + '/summaries'
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)

        super(LSTMResNet, self).__init__()

    def create_outputs(self):
        lstm_resnet_output = resnet(
            self.placeholders['images'], self.placeholders['is_training'], self.resnet_type, self.num_output_channels
        )

        self.outputs['logits'] = lstm_resnet_output

        if self.num_output_channels == 1:
            self.outputs['sigmoid'] = tf.nn.sigmoid(self.outputs['logits'])
            self.outputs['segmentation'] = self.outputs['sigmoid'] > 0.5
        else:
            self.outputs['sigmoid'] = tf.nn.softmax(self.outputs['logits'])
            self.outputs['segmentation'] = self.outputs['sigmoid'] > 0.5

        # not a dice loss, just dice coefficient
        self.outputs['dice'] = dice(self.placeholders['labels'], tf.cast(self.outputs['segmentation'], tf.float32))


class LSTMResNetEndovis(LSTMResNet, ENDOVIS2015):
    def __init__(self, jdata, data_folder, model_folder):
        LSTMResNet.__init__(self, jdata=jdata, model_folder=model_folder)
        ENDOVIS2015.__init__(self, jdata=jdata, data_folder=data_folder)


class LSTMResNetMontgomeryXray(LSTMResNet, MontgomeryXRay):
    def __init__(self, jdata, data_folder, model_folder):
        LSTMResNet.__init__(self, jdata=jdata, model_folder=model_folder)
        MontgomeryXRay.__init__(self, jdata=jdata, data_folder=data_folder)


class ResNetEndovis(ResNet, ENDOVIS2015):
    def __init__(self, jdata, data_folder, model_folder):
        ResNet.__init__(self, jdata=jdata, model_folder=model_folder)
        ENDOVIS2015.__init__(self, jdata=jdata, data_folder=data_folder)


class ResNetMontgomeryXray(ResNet, MontgomeryXRay):
    def __init__(self, jdata, data_folder, model_folder):
        ResNet.__init__(self, jdata=jdata, model_folder=model_folder)
        MontgomeryXRay.__init__(self, jdata=jdata, data_folder=data_folder)
