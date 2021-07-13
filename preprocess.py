"""Preprocessing the datasets.

Label the dataset as genuine and forged and save
preprocessed training images. The training images
are then divided into pairs, triplets or quadruplets
based on the mode of training.
"""
import os
import glob
import pickle
import random
from itertools import combinations_with_replacement
from tqdm import tqdm
import numpy as np
import cv2
from sklearn.utils import shuffle

# numpy random genrator
rng = np.random.default_rng()


# ###################################################################################################
class PreProcessing():
    """Preprocessing class.

    Attributes:
        name -- str : name of dataset.
        data_path -- str : path to dataset.
        save_path -- str : path to pickle files.
        channels -- int : channels for images, by default `1`.
        size -- int : size of images, by default `224`.
    """

    def __init__(self, name, data_path, save_path="data\\pickle-files",
                 channels=1, size=224, reset=False):

        self.SIZE = size
        self.CHANNELS = channels
        self.INPUT_SHAPE = (size, size, channels)

        self.name = name
        self.save_path = save_path
        self.f_name = '{0}{1}_'.format(name, channels)

        self.reset = reset

        # loading/making training pickle
        self.train_pickle = os.path.join(self.save_path, self.f_name+'train.pickle')

        if os.path.exists(self.train_pickle) and self.reset == False:
            with open(os.path.join(self.train_pickle), "rb") as f:
                self.train_images = pickle.load(f)
                print("\nTraining images loaded!\n")
        else:
            self.train_images = self.load_images(data_path)
            self.save_pickle(self.f_name+'train.pickle', self.train_images)

    def save_pickle(self, pickle_name, obj):
        """Save pickle file.

        Args:
            pickle_name -- str : name of pickle file to be saved.
            obj -- list/np.array : variable to be written to file.
        """
        with open(os.path.join(self.save_path, pickle_name), "wb") as f:
            pickle.dump(obj, f, protocol=4)
        print("\nPickle file saved!\n")

    def preprocess_image(self, img_path):
        """Preprocess images for better training.

        Image is converted to HSV and a mask is created
        to separate the background from the written text.
        Morphological transformations are done using OpenCV
        to achieve a well defined final input image.

        Args:
            img_path -- str : path to img.

        Returns:
            img -- np.array : processed image.
        """
        image = cv2.imread(img_path)

        # removing background and thresholding
        # for a well defined input
        result = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 0, 0])
        upper = np.array([226, 226, 226])
        mask = cv2.inRange(image, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        result[close == 0] = (255, 255, 255)

        retouch_mask = (result <= [250., 250., 250.]).all(axis=2)
        result[retouch_mask] = [0, 0, 0]

        if self.CHANNELS == 1:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # resizing and reshaping for input
        img = cv2.resize(result, (self.SIZE, self.SIZE))
        img = np.reshape(img, self.INPUT_SHAPE) / 255

        return img

    def load_images(self, data_path):
        """Load all images from dataset.

        All images are loaded, genuine and forged
        signatures are separated and processed
        using self.preprocess_image.

        Args:
            data_path -- str : path to data.

        Returns:
            data -- np.array : images with training data of
                               shape (n_classes, 2, INPUT SHAPE).
        """
        data = []
        total_data = sorted(glob.glob(f"{data_path}\\*"))
        for dirs in tqdm(total_data, desc="LOADING DATASET"):
            pos = []
            neg = []
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


# ###################################################################################################
class SiamesePairs(PreProcessing):
    """Siamese CNN pairs class for preprocessing.

    Inheriting methods and variables from PreProcessing class.

    Attributes:
        name -- str : name of dataset.
        data_path -- str : path to dataset.
        save_path -- str : path to pickle files.
        channels -- int : channels for images, by default `1`.
        size -- int : size of images, by default `224`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pair_pickle = os.path.join(
            self.save_path, self.f_name+'pairs.pickle')
        self.target_pickle = os.path.join(
            self.save_path, self.f_name+'targets.pickle')

        if os.path.exists(self.pair_pickle) and self.reset == False:
            with open(self.pair_pickle, "rb") as f:
                self.pairs = pickle.load(f)
                print("\nPairs loaded!\n")

        if os.path.exists(self.target_pickle) and self.reset == False:
            with open(self.target_pickle, "rb") as f:
                self.targets = pickle.load(f)
                print("\nTargets loaded!\n")
        else:
            self.pairs, self.targets = self.get_all_pairs()
            self.save_pickle(self.f_name+'pairs.pickle', self.pairs)
            self.save_pickle(self.f_name+'targets.pickle', self.targets)

    def get_combinations(self, length):
        """Generate list of all possible combinations.

        Args:
            length -- int : length of images.

        Returns:
            left_id -- list : list of all left ids.
            right_id -- list : list of all right ids.
            len(comb) -- int : batch_size.
        """
        comb = list(combinations_with_replacement(range(length), 2))*2
        left_id, right_id = [x[0] for x in comb], [x[1] for x in comb]

        return (left_id, right_id, len(comb))

    def get_all_pairs(self):
        """Get all pairs of images.

        Returns:
            pairs -- list : list of all pairs of images.
            targets -- np.array : targets of all pairs.
        """
        # shape = (n_classes, 2, n_images, 224, 224, 1)
        X = self.train_images

        # number of classes and images
        n_classes = X.shape[0]
        length = len(X[0][0]) // 4
        left_id, right_id, batch_size = self.get_combinations(length)

        # initialize 2 empty arrays for the input image batch
        pairs = [
            np.zeros((n_classes, batch_size, self.SIZE, self.SIZE, self.CHANNELS)) for _ in range(2)]

        # initialize vector for the targets
        targets = np.zeros((batch_size*n_classes,))
        for i in range(0, batch_size*n_classes, batch_size):
            targets[i:i+batch_size//2] = 1

        for i in tqdm(range(n_classes), desc="GETTING PAIRS"):
            for j in range(batch_size):
                # anchor
                pairs[0][i][j] = (X[i][0][left_id[j]])

                if j < batch_size//2:
                    # positive pair
                    pairs[1][i][j] = (X[i][0][right_id[j]])
                else:
                    # negative pair
                    pairs[1][i][j] = (X[i][1][left_id[j]])

        pairs = np.array(pairs).reshape(2, batch_size*n_classes,
                                        self.SIZE, self.SIZE, self.CHANNELS)

        return (pairs, targets)

    def generate_pair(self, batch_size):
        """Generator for batches.

        Args:
            batch_size -- int : batch size for model.
            train -- bool : true if training, false for validation.

        Yields:
            pairs, targets -- generator object
        """
        x = self.pairs
        y = self.targets
        batch_size = min(batch_size, y.shape[0])

        indices = np.arange(y.shape[0])
        np.random.shuffle(indices)
        x, y = x[:, indices], y[indices]

        for i in range(batch_size):
            pair = [x[0][i], x[1][i]]
            target = y[i]
            yield (pair, target)


# ###################################################################################################
class SiameseTriplets(PreProcessing):
    """Siamese CNN triplets class for preprocessing.

    Inheriting methods and variables from PreProcessing class.

    Attributes:
        name -- str : name of dataset.
        data_path -- str : path to dataset.
        save_path -- str : path to pickle files.
        channels -- int : channels for images, by default `1`.
        size -- int : size of images, by default `224`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.triplet_pickle = os.path.join(
            self.save_path, self.f_name+'triplets.pickle')

        if os.path.exists(self.triplet_pickle) and self.reset == False:
            with open(self.triplet_pickle, "rb") as f:
                self.triplets = pickle.load(f)
                print("\nTriplets loaded!\n")
        else:
            n_classes = self.train_images.shape[0]
            self.triplets = self.get_all_triplets(n_classes*12)
            self.save_pickle(self.f_name+'triplets.pickle', self.triplets)

    def get_index(self, p_len, n_len):
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

    def get_all_triplets(self, batch_size):
        """Get batches.

        Args:
            batch_size -- int : batch size for model.
            train -- bool : true if training, false for validation.

        Returns:
            triplets -- list : list of triplets.
        """
        X = self.train_images

        # shape = (n_classes, 2, n_images, 224, 224, 1)
        n_classes = X.shape[0]
        p_len = len(X[0][0])
        n_len = len(X[0][1])
        # batch_size = min(n_classes, batch_size)*2

        # initialize 2 empty arrays for the input image batch
        triplets = [
            np.zeros((batch_size, self.SIZE, self.SIZE, self.CHANNELS)) for _ in range(3)]

        c = 0
        for i in tqdm(range(batch_size), desc="LOADING TRIPLETS"):
            a, p, n = self.get_index(p_len, n_len)
            if c >= n_classes:
                c = 0

            triplets[0][i, :, :, :] = X[c][0][a]
            triplets[1][i, :, :, :] = X[c][0][p]
            triplets[2][i, :, :, :] = X[c][1][n]

            c += 1

        return triplets

    def generate_triplet(self, batch_size):
        """Generator for batches.

        Args:
            batch_size -- int : batch size for model.
            train -- bool : true if training, false for validation.

        Yields:
            triplets -- generator object
        """
        x = self.triplets
        batch_size = min(batch_size, x.shape[1])

        indices = np.arange(x.shape[1])
        np.random.shuffle(indices)
        x = x[:, indices]

        for i in range(batch_size):
            triplet = [x[0][i], x[1][i], x[2][i]]
            yield triplet


# ###################################################################################################
class SiameseQuadruplets(PreProcessing):
    """Siamese CNN quadruplets class for preprocessing.

    Inheriting methods and variables from PreProcessing class.

    Args:
        name -- str : name of dataset.
        data_path -- str : path to dataset.
        save_path -- str : path to pickle files.
        channels -- int : channels for images, by default `1`.
        size -- int : size of images, by default `224`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.quadruplet_pickle = os.path.join(
            self.save_path, self.f_name+'quadruplets.pickle')

        if os.path.exists(self.quadruplet_pickle) and self.reset == False:
            with open(self.quadruplet_pickle, "rb") as f:
                self.quadruplets = pickle.load(f)
                print("\nQuadruplets loaded!\n")
        else:
            n_classes = self.train_images.shape[0]
            self.quadruplets = self.get_all_quadruplets(n_classes*12)
            self.save_pickle(self.f_name+'quadruplets.pickle',
                             self.quadruplets)

    def get_index(self, p_len, n_len):
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

    def get_all_quadruplets(self, batch_size):
        """Get batches.

        Args:
            batch_size -- int : batch size for model.
            train -- bool : true if training, false for validation.

        Returns:
            quadruplets -- list : list of pairs.
        """
        X = self.train_images

        # shape = (n_classes, 2, n_images, 224, 224, 1)
        n_classes = X.shape[0]
        p_len = len(X[0][0])
        n_len = len(X[0][1])
        # batch_size = min(n_classes, batch_size)*2

        # initialize 2 empty arrays for the input image batch
        quadruplets = [
            np.zeros((batch_size, self.SIZE, self.SIZE, self.CHANNELS)) for _ in range(4)]

        c = 0
        for i in tqdm(range(batch_size), desc="LOADING QUADRUPLETS"):
            c2 = random.randint(0, n_classes-1)
            while c2 == c:
                c2 = random.randint(0, n_classes-1)
            a, p, n = self.get_index(p_len, n_len)
            if c >= n_classes:
                c = 0

            quadruplets[0][i, :, :, :] = X[c][0][a]
            quadruplets[1][i, :, :, :] = X[c][0][p]
            quadruplets[2][i, :, :, :] = X[c][1][n]
            quadruplets[3][i, :, :, :] = X[c2][1][a]

            c += 1

        return quadruplets

    def generate_quadruplet(self, batch_size):
        """Generator for batches.

        Args:
            batch_size -- int : batch size for model.

        Yields:
            quadruplet -- generator object
        """
        x = self.quadruplets
        batch_size = min(batch_size, x.shape[1])

        indices = np.arange(x.shape[1])
        np.random.shuffle(indices)
        x = x[:, indices]

        for i in range(batch_size):
            quadruplet = [x[0][i], x[1][i], x[2][i], x[3][i]]
            yield quadruplet
