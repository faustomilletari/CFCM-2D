import os

import imageio
import numpy as np
from skimage import io, color, morphology


def load_ISBI_tif_transform(tif_image_path, tif_label_path=None):
    def transform(data):
        data['images'] = io.imread(tif_image_path)

        if tif_label_path is not None:
            data['labels'] = io.imread(tif_label_path)
        else:
            data['labels'] = None

        return data

    return transform


def readFrames(video_path, grayscale=False):
    # load video
    vid = imageio.get_reader(video_path, 'ffmpeg')
    # load all frames
    frames = list(vid)
    if grayscale:
        frames = [color.rgb2gray(f) for f in frames]
        frames = [f[:, :, np.newaxis] for f in frames]
    images = np.array(frames)

    return images


def readGTFrames(video_path):
    directory, _ = os.path.split(video_path)

    if directory.endswith('2') or directory.endswith('3') or directory.endswith('4'):
        # load video
        vid = imageio.get_reader(video_path, 'ffmpeg')
        # load all frames
        frames = list(vid)
        frames = [f[:, :, 0] for f in frames]  # get only one channel

    if directory.endswith('1'):
        if 'Training' in directory:
            right_tool_video = os.path.join(directory, 'Segmentation.avi')
            vid = imageio.get_reader(right_tool_video, 'ffmpeg')
            frames = list(vid)
            frames = [f[:, :, 0] for f in frames]  # get only one channel
        else:
            right_tool_video = os.path.join(directory, 'Right_Instrument_Segmentation.avi')
            vid = imageio.get_reader(right_tool_video, 'ffmpeg')
            frames = list(vid)
            frames = [f[:, :, 0] for f in frames]  # get only one channel

        left_tool_video = os.path.join(directory, 'Left_Instrument_Segmentation.avi')
        vid2 = imageio.get_reader(left_tool_video, 'ffmpeg')
        # load all frames of left tool and add them to upper frames
        frames_left = list(vid2)
        frames_left = [f[:, :, 0] for f in frames_left]  # get only one channel
        frames = [f1 + f2 for f1, f2 in zip(frames, frames_left)]

    if directory.endswith('5'):
        right_tool_video = os.path.join(directory, 'Right_Instrument_Segmentation.avi')
        vid = imageio.get_reader(right_tool_video, 'ffmpeg')
        frames = list(vid)
        frames = [f[:, :, 0] for f in frames]  # get only one channel

        left_tool_video = os.path.join(directory, 'Left_Instrument_Segmentation.avi')
        vid2 = imageio.get_reader(left_tool_video, 'ffmpeg')
        # load all frames of left tool and add them to upper frames
        frames_left = list(vid2)
        frames_left = [f[:, :, 0] for f in frames_left]  # get only one channel
        frames = [f1 + f2 for f1, f2 in zip(frames, frames_left)]

        frames = frames[1:]

    if directory.endswith('6'):
        right_tool_video = os.path.join(directory, 'Segmentation.avi')
        vid = imageio.get_reader(right_tool_video, 'ffmpeg')
        frames = list(vid)
        frames = [f[:, :, 0] for f in frames]  # get only one channel
        frames = frames[1:]

    # modify labels (attention! compression artefacts)
    # original:
    # background: 0 , Grasper: 70, Shaft: 160
    # new:
    # background: 0 , Grasper: 1, Shaft: 2
    def relabel(f):
        f[f < 10] = 0
        f[f > 90] = 2
        f[f > 2] = 1
        f = morphology.opening(f)
        return f

    frames = [relabel(f) for f in frames]
    images = np.array(frames)
    return images


def load_EndoVis_transform(videos_paths, label_path=None):
    # video_paths: list of references to considered datasets
    # e.g. ['Training/Dataset1/Video.avi','Training/Dataset2/Video.avi','Training/Dataset4/Video.avi']

    def transform(data):
        print('-' * 30)
        print('Load Data')
        data['images'] = []
        data['labels'] = []

        data['images'] = np.vstack([readFrames(v) for v in videos_paths])

        if label_path is not None:
            data['labels'] = (np.vstack([readGTFrames(v) for v in label_path])).astype(np.float32)
        else:
            data['labels'] = None

        print('-' * 30)
        return data

    return transform


def load_montgomery_xray(image_path_list, label_left_path_list, label_right_path_list):
    def transform(data):
        assert len(image_path_list) == len(label_left_path_list)
        assert len(label_left_path_list) == len(label_right_path_list)

        images = []
        labels = []
        for image_file, left_file, right_file in zip(image_path_list, label_left_path_list, label_right_path_list):
            image = io.imread(image_file, as_grey=True).astype(np.float32)
            ll = io.imread(left_file, as_grey=True).astype(np.float32)
            lr = io.imread(right_file, as_grey=True).astype(np.float32)

            label = np.zeros_like(image)
            label[ll > 0] = 1
            label[lr > 0] = 2

            images.append(image)
            labels.append(label)

        data['images'] = images
        data['labels'] = labels

        return data

    return transform
