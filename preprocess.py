"""Preprocessing.

1. Label the dataset.
2. Preprocess the images.
3. Make batches of triplets with anchor, positive and negative images.
"""
import os
import glob
import pickle
import random
from tqdm import tqdm
import numpy as np
import cv2


class PreProcessing():
    """Preprocessing class.
    """

    def __init__(self, data_path):
        self.SIZE = 224
        self.INPUT_SHAPE = (self.SIZE, self.SIZE, 1)

        self.data_path = data_path

        # loading/making training and validation pickle
        self.train_pickle = os.path.join(self.data_path, 'train.pickle')
        self.val_pickle = os.path.join(self.data_path, 'validation.pickle')

        if os.path.exists(self.train_pickle) and os.path.exists(self.val_pickle):
            with open(os.path.join(self.train_pickle), "rb") as f:
                self.train_images = pickle.load(f)
            with open(os.path.join(self.val_pickle), "rb") as f:
                self.val_images = pickle.load(f)
        else:
            self.train_images, self.val_images = self.load_images(data_path)
            self.save_pickle('train.pickle', self.train_images)
            self.save_pickle('validation.pickle', self.val_images)

    def save_pickle(self, pickle_name, obj):
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
        train_len = int(data.shape[0]*0.8)

        return data[:train_len], data[train_len:]

    def get_pair(self, p_len, n_len):
        """Get image pairs.

        Args:
            p_len -- int : length of positive images.
            n_len -- int : length of negative images.

        Returns:
            p1 -- int : index for positive image.
            p2 -- int : index for matching image.
            n -- int : index for negative image.
        """
        p1 = random.randint(0, p_len-1)
        p2 = random.randint(0, p_len-1)
        n = random.randint(0, n_len-1)

        return p1, p2, n

    def get_all_batches(self, batch_size, train=True):
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
