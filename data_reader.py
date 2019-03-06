import os
import scipy.signal
import sklearn.utils
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold

def read_data(input_size, output_size, x_range, y_range, cross_val=5, val_fold=0, batch_size=100,
                 shuffle_size=100, data_dir=os.path.dirname(__file__), rand_seed=1234):
    """
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
    """
    Read feature and label
    :param is_train: the dataset is used for training or not
    :param train_valid_tuple: if it's not none, it will be the names of train and valid files
    :return: feature and label read from csv files, one line each time
    """

    # get data files
    def importData(directory):
            # pull data into python, should be either for training set or eval set
        train_data_files = []
        for file in os.listdir(os.path.join(directory)):
            if file.endswith('.csv'):
                train_data_files.append(file)
        print(train_data_files)
        # get data
        ftr = []
        lbl = []
        for file_name in train_data_files:
            # import full arrays
            ftr_array = pd.read_csv(os.path.join(directory, file_name), delimiter=',', usecols=x_range)
            lbl_array = pd.read_csv(os.path.join(directory, file_name), delimiter=',', usecols=y_range)
            # append each data point to ftr and lbl
            for params, curve in zip(ftr_array.values, lbl_array.values):
                ftr.append(params)
                lbl.append(curve)
        ftr = np.array(ftr, dtype='float32')
        lbl = np.array(lbl, dtype='float32')
        return ftr, lbl
    print('getting data files')

    ftrTrain, lblTrain = importData(os.path.join(data_dir, 'dataIn'))
    ftrTest, lblTest = importData(os.path.join(data_dir, 'dataIn', 'eval'))

    print('total number of training samples is {}'.format(len(ftrTrain)))
    print('total number of test samples is {}'.format(len(ftrTest)))

    print('downsampling output curves')
    # resample the output curves so that there are not so many output points
    lblTrain = scipy.signal.resample(lblTrain, output_size+50, axis=1)
    lblTest = scipy.signal.resample(lblTest, output_size+50, axis=1)

    # remove the ringing that occurs on the end of the spectra due to the Fourier method used by scipy
    lblTrain = np.array([spec[25:-25] for spec in lblTrain])
    lblTest = np.array([spec[25:-25] for spec in lblTest])

    # determine lengths of training and validation sets
    num_data_points = len(ftrTrain)
    train_length = int(.8 * num_data_points)

    print('generating TF dataset')
    assert np.shape(ftrTrain)[0] == np.shape(lblTrain)[0]
    assert np.shape(ftrTest)[0] == np.shape(lblTest)[0]

    dataset_train = tf.data.Dataset.from_tensor_slices((ftrTrain, lblTrain))
    dataset_valid = tf.data.Dataset.from_tensor_slices((ftrTest, lblTest))
    # shuffle then split into training and validation sets
    dataset_train = dataset_train.shuffle(shuffle_size)

    dataset_train = dataset_train.repeat()
    dataset_train = dataset_train.batch(batch_size, drop_remainder=True)
    dataset_valid = dataset_valid.batch(batch_size, drop_remainder=True)

    iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    features, labels = iterator.get_next()
    train_init_op = iterator.make_initializer(dataset_train)
    valid_init_op = iterator.make_initializer(dataset_valid)

    return features, labels, train_init_op, valid_init_op

if __name__ == '__main__':
    print('testing read_data')
    read_data(input_size=2,
              output_size=300,
              x_range=[i for i in range(2, 10)],
              y_range=[i for i in range(10, 2011)],
              cross_val=5,
              val_fold=0,
              batch_size=100,
              shuffle_size=100,
              data_dir=os.path.dirname(__file__), rand_seed=1234)
    print('done.')