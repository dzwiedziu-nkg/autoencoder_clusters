import pickle
from math import ceil

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
from skimage import filters
from skimage.measure import regionprops


DOTS_DST = 'cache/dots_v2.pickle'
TRACKS_DST = 'cache/tracks_v2.pickle'
WORMS_DST = 'cache/worms_v2.pickle'
ARTIFACTS_DST = 'cache/artifacts_v2.pickle'


def load_from_file(fn):
    return pickle.loads(open(fn, "rb").read())


def load_data(use_dots=False, use_tracks=False, use_worms=False, use_artefacts=False, cut_to_mnist=True):
    w, h = 60, 60
    w1 = int((w - 28) / 2)
    h1 = int((h - 28) / 2)

    x_test_stack = []
    y_test_stack = []
    x_train_stack = []
    y_train_stack = []

    def append_stack(pickle_fn, no):
        test, train = load_from_file(pickle_fn)
        if cut_to_mnist:
            x_test_stack.append(test[:, w1:w1 + 28, h1:h1 + 28] * 255)
            x_train_stack.append(train[:, w1:w1 + 28, h1:h1 + 28] * 255)
        else:
            x_test_stack.append(test[:, :, :] * 255)
            x_train_stack.append(train[:, :, :] * 255)

        y_test = np.ones(test.shape[0]) * no
        y_train = np.ones(train.shape[0]) * no

        y_test_stack.append(y_test)
        y_train_stack.append(y_train)

    if use_dots:
        append_stack(DOTS_DST, 1)
    if use_tracks:
        append_stack(TRACKS_DST, 2)
    if use_worms:
        append_stack(WORMS_DST, 3)
    if use_artefacts:
        append_stack(ARTIFACTS_DST, 4)

    x_test = np.vstack(x_test_stack)
    x_train = np.vstack(x_train_stack)

    y_test = np.hstack(y_test_stack)
    y_train = np.hstack(y_train_stack)
    return (x_train, y_train), (x_test, y_test)


def plot_images(images, start_i=0, max_images=None):
    cols = 100
    rows = 10
    imgs = cols * rows

    if max_images is None:
        max_images = len(images) - start_i
    to_plot = min(max_images, len(images) - start_i)

    if to_plot > imgs:
        ploted = 0
        while ploted < to_plot:
            plot_images(images, start_i + ploted, imgs)
            ploted += imgs
        return

    cols = min(cols, ceil(to_plot / rows))

    plt.figure(figsize=(rows * 2, cols * 2))
    for i in range(0, to_plot):
        ax = plt.subplot(cols, rows, i + 1)
        plt.imshow(images[start_i + i].reshape(60, 60))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def normalize_rotation(image):
    values = []

    img = image.astype(np.uint8)
    for j in range(0, 180):
        img2 = rotate(img, j, reshape=False)
        sums2 = np.sum(img2, axis=1)
        values.append([j, np.max(sums2)])
    s4 = sorted(values, key=lambda x: (-x[1]))
    return rotate(img, s4[0][0], reshape=False)


def normalize_translation(image):
    threshold_value = filters.threshold_otsu(image)
    labeled_foreground = (image > threshold_value).astype(int)
    properties = regionprops(labeled_foreground, image)

    rolled = np.roll(image, round(30-properties[0].centroid[0]), axis=0)
    rolled = np.roll(rolled, round(30-properties[0].centroid[1]), axis=1)
    return rolled