import numpy as np
from skimage.filters import gaussian
from skimage.transform import resize


def convert_to_float():
    def transform(data):
        data['images'] = data['images'].astype(np.float32)

        return data

    return transform


def shuffle_data(keys=None):
    def transform(data):
        if keys is not None:
            data_keys = keys
        else:
            data_keys = data.keys()
        data_length = len(data[data_keys[0]])
        new_order = np.random.permutation(data_length)

        for key in data_keys:
            data[key] = data[key][new_order]

        return data

    return transform


def add_trailing_singleton_dimension_transform(field):
    def transform(data):
        data[field] = data[field][..., np.newaxis]

        return data

    return transform


def threshold_labels_transform():
    def transform(data):
        data['labels'] = (data['labels'] > 0.5).astype(np.float32)

        return data

    return transform


def resize_img(size):
    def transform(data):
        data["images"] = np.array(
            [resize(i, (size, size), preserve_range=True) for i in data['images']],
            dtype=data['images'][0].dtype
        )
        data['labels'] = np.array(
            [resize(i, (size, size), preserve_range=True, order=0) for i in data['labels']],
            dtype=data['labels'][0].dtype
        )
        return data

    return transform


def flip_left_right(probability=0.5):
    def transform(data):
        for i in range(len(data['images'])):
            if np.random.rand() > probability:
                data['images'][i] = np.fliplr(data['images'][i])
                data['labels'][i] = np.fliplr(data['labels'][i])
        return data

    return transform


def flip_up_down(probability=0.5):
    def transform(data):
        for i in range(len(data['images'])):
            if np.random.rand() > probability:
                data['images'][i] = np.flipud(data['images'][i])
                data['labels'][i] = np.flipud(data['labels'][i])
        return data

    return transform


def normalize_between_zero_and_one():
    def transform(data):
        images_t = []
        for image in data['images']:
            if np.max(image) > 1.0:
                image /= 255.0
            images_t.append(image)

        data['images'] = np.stack(images_t)
        return data

    return transform


def specularity(probability=0.5):
    def transform(data):
        def specnoise(image, segmentation):

            radius = np.random.randint(1, 8) / 100 * image.shape[0]
            blur = np.random.randint(1, 5) / 100
            # convert image to [0,1]
            originalDataType = image.dtype

            if image.dtype == ('uint8'):
                image = image.astype(dtype=np.float32) / 255.0

            # get random position on valid area
            if len(segmentation.shape) == 3:
                px, py, _ = np.where(segmentation > 0)
            else:
                px, py = np.where(segmentation > 0)

            # exception if no valid area is possible
            if px.size == 0:
                return image

            i = np.random.randint(len(px))
            random_pos = [px[i], py[i]]

            # create noise mask
            x, y = np.indices((image.shape[0], image.shape[1]))
            noise = gaussian((x - random_pos[0]) ** 2 + (y - random_pos[1]) ** 2 < radius ** 2,
                             sigma=blur * image.shape[0])

            # apply on image, but make sure that it is only on the valid area
            if len(image.shape) == 3:
                image = image + np.stack((noise * (segmentation[:, :, 0] > 0),) * 3, -1)
            else:
                image = image + np.stack((noise * (segmentation > 0),) * 3, -1)

            # cap too high values
            image[image > 1] = 1

            # convert back to original format
            if image.dtype == ('uint8'):
                image = (image * 255.0).astype(dtype=np.uint8)

            return image

        for i in range(len(data['images'])):
            if np.random.rand() > probability:
                # add noise only on instrument
                data['images'][i] = specnoise(data['images'][i], data['labels'][i])
                # add noise only on background
                data['images'][i] = specnoise(data['images'][i], (data['labels'][i] < 1))
        return data

    return transform


def crop_actual_image():
    def transform(data):
        images_t = []
        labels_t = []

        for image, label in zip(data['images'], data['labels']):
            non_zero_x = np.where(np.sum(image, axis=0) > 0)
            non_zero_y = np.where(np.sum(image, axis=0) > 0)

            image_t = image[np.min(non_zero_x):np.max(non_zero_x), np.min(non_zero_y):np.max(non_zero_y)]
            label_t = label[np.min(non_zero_x):np.max(non_zero_x), np.min(non_zero_y):np.max(non_zero_y)]

            images_t.append(image_t)
            labels_t.append(label_t)

        data['images'] = images_t
        data['labels'] = labels_t
        return data

    return transform


def make_onehot(num_classes):
    def transform(data):
        labels_t = []

        for label in data['labels']:
            label_t = np.zeros((label.shape[0], label.shape[1], num_classes), dtype=np.float32)
            for i in range(num_classes):
                label_t[:, :, i] = (label == i).astype(np.float32)

            labels_t.append(label_t)

        data['labels'] = np.stack(labels_t)

        return data

    return transform
