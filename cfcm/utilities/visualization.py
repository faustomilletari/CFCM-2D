import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import tfplot
import numpy as np
from scipy.ndimage import morphology

from textwrap import wrap
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc


class ConfusionMatrix(object):
    def __init__(self, labels, normalize=False):
        self.labels = labels
        self.normalize = normalize

        self.true_labels = np.empty(shape=(0,))
        self.predicted_labels = np.empty(shape=(0,))

    def __call__(self, true_labels, predict_labels):
        self.true_labels = np.concatenate([self.true_labels, true_labels.astype(int)])
        self.predicted_labels = np.concatenate([self.predicted_labels, predict_labels.astype(int)])

    def plot(self):
        cm = confusion_matrix(self.true_labels, self.predicted_labels)
        if self.normalize:
            cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm, copy=True)
            cm = cm.astype('int')

        np.set_printoptions(precision=2)

        fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(cm, cmap='Oranges')

        classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in self.labels]
        classes = ['\n'.join(wrap(l, 40)) for l in classes]

        tick_marks = np.arange(len(classes))

        ax.set_xlabel('Predicted', fontsize=7)
        ax.set_xticks(tick_marks)
        c = ax.set_xticklabels(classes, fontsize=10, rotation=-90, ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label', fontsize=7)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=10, va='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.',
                    horizontalalignment="center", fontsize=20,
                    verticalalignment='center', color="black"
                    )
        fig.set_tight_layout(True)
        array = tfplot.figure.to_array(fig)
        plt.close('all')
        return array


class PrecisionRecallCurve(object):
    def __init__(self, normalize=False):
        plt.close('all')
        self.normalize = normalize

        self.true_labels = np.empty(shape=(0,))
        self.predicted_labels = np.empty(shape=(0,))

    def __call__(self, true_labels, predict_labels):
        self.true_labels = np.concatenate([self.true_labels, true_labels.astype(int)])
        self.predicted_labels = np.concatenate([self.predicted_labels, predict_labels.astype(float)])

    def plot(self):
        average_precision = average_precision_score(self.true_labels, self.predicted_labels)

        precision, recall, _ = precision_recall_curve(self.true_labels, self.predicted_labels)

        fig = plt.figure()

        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
        fig.set_tight_layout(True)
        array = tfplot.figure.to_array(fig)

        return array


class RoCCurve(object):
    def __init__(self, normalize=False):
        self.normalize = normalize

        self.true_labels = np.empty(shape=(0,))
        self.predicted_labels = np.empty(shape=(0,))

    def __call__(self, true_labels, predict_labels):
        self.true_labels = np.concatenate([self.true_labels, true_labels.astype(int)])
        self.predicted_labels = np.concatenate([self.predicted_labels, predict_labels.astype(float)])

    def plot(self):
        fpr, tpr, _ = roc_curve(self.true_labels, self.predicted_labels)
        roc_auc = auc(fpr, tpr)

        fig = plt.figure()

        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()
        fig.set_tight_layout(True)
        array = tfplot.figure.to_array(fig)
        plt.close('all')
        return array


def surfd(input1, input2, sampling=1, connectivity=1):
    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = input_1 ^ morphology.binary_erosion(input_1, conn)
    Sprime = input_2 ^ morphology.binary_erosion(input_2, conn)

    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

    return sds
