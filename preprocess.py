"""Preprocessing.

1. Label the dataset.
2. Preprocess the images.
3. Make labeled pairs of images.

"""
import os
import glob
import pickle
import random
from itertools import combinations_with_replacement
from tqdm import tqdm
import numpy as np
import cv2

# numpy random genrator
rng = np.random.default_rng()


class PreProcessing():
    """Preprocessing class.
    """

    def __init__(self, name, data_path, size):
        self.SIZE = size
        self.INPUT_SHAPE = (self.SIZE, self.SIZE, 1)

        self.name = name
        self.data_path = data_path

        # loading/making training and validation pickle
        self.train_pickle = os.path.join(self.data_path, f'{name}_train.pickle')

        if os.path.exists(self.train_pickle):
            with open(os.path.join(self.train_pickle), "rb") as f:
                self.train_images = pickle.load(f)
        else:
            self.train_images = self.load_images(data_path)
            self.save_pickle(f'{name}_train.pickle', self.train_images)

        # pairs and targets
        self.pair_pickle = os.path.join(self.data_path, f'{name}_pairs.pickle')
        self.target_pickle = os.path.join(self.data_path, f'{name}_targets.pickle')

        if os.path.exists(self.pair_pickle) and os.path.exists(self.target_pickle):
            with open(os.path.join(self.pair_pickle), "rb") as f:
                self.pairs = pickle.load(f)
            with open(os.path.join(self.target_pickle), "rb") as f:
                self.targets = pickle.load(f)
        else:
            self.pairs, self.targets = self.get_all_batches()
            self.save_pickle(f'{name}_pairs.pickle', self.pairs)
            self.save_pickle(f'{name}_targets.pickle', self.targets)

    def save_pickle(self, pickle_name, obj):
        """Save pickle file.

        Args:
            pickle_name -- str : name of pickle file to be saved.
            obj -- list/np.array : variable to be written.
        """
        with open(os.path.join(self.data_path, pickle_name), "wb") as f:
            pickle.dump(obj, f)

    def preprocess_image(self, img_path):
        """Preprocess images.

        Args:
            img_path -- str : path to img.

        Returns:
            img -- np.array : processed image.
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.SIZE, self.SIZE))
        img = np.reshape(img, self.INPUT_SHAPE) / 255

        return img

    def load_images(self, data_path):
        """Load images.

        Args:
            data_path -- str : path to data.

        Returns:
            training data -- np.array : images with training data.
            validation data -- np.array : images with validation data.
        """
        data = []
        total_data = sorted(glob.glob(f"{data_path}\\*"))
        for dirs in tqdm(total_data):
            pos = []
            neg = []
            if '.pickle' not in dirs:
                for img_path in sorted(glob.glob(f"{dirs}\\*")):
                    img = self.preprocess_image(img_path)

                    # appending image
                    if '-G-' in img_path or 'original' in img_path:
                        pos.append(img)
                    else:
                        neg.append(img)

            data.append([pos, neg])

        data = np.array(data)
        return data

    def get_all_batches(self):
        """Get all pairs of images.

        Args:
            p_len -- int : length of positive images.
            n_len -- int : length of negative images.

        Returns:
            pairs -- list : list of all pairs of images.
            targets -- np.array : targets of all pairs.
        """
        # shape = (n_classes, 2, n_images, 224, 224, 1)
        X = self.train_images

        n_classes = X.shape[0]
        p_len = len(X[0][0])

        comb = list(combinations_with_replacement(range(p_len), 2))*2
        batch_size = len(comb)
        left_id, right_id = [x[0] for x in comb], [x[1] for x in comb]

        # initialize 2 empty arrays for the input image batch
        pairs = [np.zeros((batch_size, self.SIZE, self.SIZE, 1))
                 for _ in range(2)]

        # initialize vector for the targets
        targets = np.zeros((batch_size,))
        targets[:batch_size//2] = 1

        for i in tqdm(range(n_classes)):
            for j in range(batch_size):
                # anchor
                pairs[0][j] = X[i][0][left_id[j]]

                if j < batch_size//2:
                    # positive pair
                    pairs[1][j] = X[i][0][right_id[j]]
                else:
                    # negative pair
                    pairs[1][j] = X[i][1][left_id[j]]

        return pairs, targets

    def get_random_pair(self, p_len, n_len):
        """Get image pairs.

        Args:
            p_len -- int : length of positive images.
            n_len -- int : length of negative images.

        Returns:
            a -- int : index for anchor image.
            p -- int : index for positive image.
            n -- int : index for negative image.
        """
        a, p = rng.integers(0, p_len-1, size=(2,))
        n = rng.integers(0, n_len-1)

        return a, p, n

    def get_random_batch(self, batch_size, train=True):
        """Get batches.

        Args:
            batch_size -- int : batch size for model.
            train -- bool : true if training, false for validation.

        Returns:
            pairs -- list : list of pairs.
            targets -- np.array : array containing target labels.
        """
        if train:
            X = self.train_images
        else:
            X = self.val_images

        # shape = (n_classes, 2, n_images, 224, 224, 1)
        n_classes = X.shape[0]
        p_len = len(X[0][0])
        n_len = len(X[0][1])
        batch_size = min(n_classes, batch_size)*2

        # initialize 2 empty arrays for the input image batch
        pairs = [np.zeros((batch_size, self.SIZE, self.SIZE, 1))
                 for i in range(2)]

        # initialize vector for the targets
        targets = np.zeros((batch_size,))
        targets[::2, ] = 1

        for i in range(0, batch_size, 2):
            c = random.randint(0, n_classes-1)
            p1, p2, n = self.get_pair(p_len, n_len)

            pairs[0][i, :, :, :] = X[c][0][p1]
            pairs[0][i+1, :, :, :] = X[c][0][p1]
            pairs[1][i, :, :, :] = X[c][0][p2]
            pairs[1][i+1, :, :, :] = X[c][1][n]

        return (pairs, targets)

    def generate_batch(self, batch_size, train=True):
        """Generator for batches.

        Args:
            batch_size -- int : batch size for model.
            train -- bool : true if training, false for validation.

        Returns:
            pairs, targets -- generator object
        """
        while True:
            pairs, targets = self.get_batches(batch_size, train)
            yield (pairs, targets)
