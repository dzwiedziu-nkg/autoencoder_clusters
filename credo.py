import pickle
import numpy as np
from sklearn.preprocessing import normalize


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
