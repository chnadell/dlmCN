import os
import scipy.signal
import sklearn.utils
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold


class DataReader(object):
    def __init__(self, input_size, output_size, x_range, y_range, cross_val=5, val_fold=0, batch_size=100,
                 shuffle_size=100, data_dir=os.path.dirname(__file__), rand_seed=1234):
        """
        Initialize a data reader
        :param input_size: input size of the arrays
        :param output_size: output size of the arrays
        :param x_range: columns of input data in the txt file
        :param y_range: columns of output data in the txt file
        :param cross_val: number of cross validation folds
        :param val_fold: which fold to be used for validation
        :param batch_size: size of the batch read every time
        :param shuffle_size: size of the batch when shuffle the dataset
        :param data_dir: parent directory of where the data is stored, by default it's the current directory
        :param rand_seed: random seed
        """
        self.input_size = input_size
        self.output_size = output_size
        self.x_range = x_range
        self.y_range = y_range
        self.cross_val = cross_val
        self.val_fold = val_fold
        self.batch_size = batch_size
        self.shuffle_zie = shuffle_size
        self.data_dir = data_dir
        np.random.seed(rand_seed)

    def data_reader(self, is_train, train_valid_tuple):
        """
        Read feature and label
        :param is_train: the dataset is used for training or not
        :param train_valid_tuple: if it's not none, it will be the names of train and valid files
        :return: feature and label read from csv files, one line each time
        """
        if not train_valid_tuple:
            data_file = os.path.join(self.data_dir, 'data', 'UnitCellData_V7.txt')
            x = np.loadtxt(data_file, delimiter=',', usecols=self.x_range)
            y = np.loadtxt(data_file, delimiter=',', usecols=self.y_range)
            y = scipy.signal.resample(y, self.output_size, axis=1)
            (x, y) = sklearn.utils.shuffle(x, y, random_state=0)
            kf = KFold(n_splits=self.cross_val)
            for cnt, (train_idx, valid_idx) in enumerate(kf.split(x)):
                if cnt == self.val_fold:
                    if is_train:
                        ftr, lbl = x[train_idx, :], y[train_idx, :]
                    else:
                        valid_num = valid_idx.shape[0] // self.batch_size * self.batch_size
                        ftr, lbl = x[valid_idx[:valid_num], :], y[valid_idx[:valid_num], :]
                    for (f, l) in zip(ftr, lbl):
                        yield f, l
        else:
            train_data_file = os.path.join(self.data_dir, 'data', train_valid_tuple[0])
            valid_data_file = os.path.join(self.data_dir, 'data', train_valid_tuple[1])
            if is_train:
                ftr = np.loadtxt(train_data_file, delimiter=',', usecols=self.x_range)
                lbl = np.loadtxt(train_data_file, delimiter=',', usecols=self.y_range)
            else:
                ftr = np.loadtxt(valid_data_file, delimiter=',', usecols=self.x_range)
                lbl = np.loadtxt(valid_data_file, delimiter=',', usecols=self.y_range)
            lbl = scipy.signal.resample(lbl, self.output_size, axis=1)
            for (f, l) in zip(ftr, lbl):
                yield f, l

    def get_dataset(self, train_valid_tuple):
        """
        Create a tf.Dataset from the generator defined
        :param train_valid_tuple: if it's not none, it will be the names of train and valid files
        :return: a tf.Dataset object
        """
        def generator_train(): return self.data_reader(True, train_valid_tuple)

        def generator_valid(): return self.data_reader(False, train_valid_tuple)

        dataset_train = tf.data.Dataset.from_generator(generator_train, (tf.float32, tf.float32),
                                              (self.input_size, self.output_size))
        dataset_valid = tf.data.Dataset.from_generator(generator_valid, (tf.float32, tf.float32),
                                                       (self.input_size, self.output_size))
        return dataset_train, dataset_valid

    def get_data_holder_and_init_op(self, train_valid_tuple=None):
        """
        Get tf iterator as well as init operation for both training and validation
        :param train_valid_tuple: if it's not none, it will be the names of train and valid files
        :return: features, labels, training init operation, validation init operation
        """
        dataset_train, dataset_valid = self.get_dataset(train_valid_tuple)
        dataset_train = dataset_train.shuffle(self.shuffle_zie)
        dataset_train = dataset_train.repeat()
        dataset_train = dataset_train.batch(self.batch_size)
        dataset_valid = dataset_valid.batch(self.batch_size)

        iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
        features, labels = iterator.get_next()
        train_init_op = iterator.make_initializer(dataset_train)
        valid_init_op = iterator.make_initializer(dataset_valid)

        return features, labels, train_init_op, valid_init_op
