import os
import tensorflow as tf
import numpy as np
import time

import utils
import network_maker
import network_helper

def gen_data(out_path, param_bounds, spacings):
    scan = []
    for h1 in np.arange(param_bounds[0, 0], param_bounds[0, 1], spacings[0]):
        scan.append(h1)
    print('possible h1 values are in {}'.format(scan))
    print('if all bounds and spacings are the same, number of combos is {}'.format(len(scan)**8))
    start = time.time()
    with open(os.path.join(out_path, 'grid.csv'), 'w+') as gfile:
        for h1 in np.arange(param_bounds[0, 0], param_bounds[0, 1], spacings[0]):
            for h2 in np.arange(param_bounds[1, 0], param_bounds[1, 1], spacings[1]):
                for h3 in np.arange(param_bounds[2, 0], param_bounds[2, 1], spacings[2]):
                    for h4 in np.arange(param_bounds[3, 0], param_bounds[3, 1], spacings[3]):
                        for r1 in np.arange(param_bounds[4, 0], param_bounds[4, 1], spacings[4]):
                            for r2 in np.arange(param_bounds[5, 0], param_bounds[5, 1], spacings[5]):
                                for r3 in np.arange(param_bounds[6, 0], param_bounds[6, 1], spacings[6]):
                                    for r4 in np.arange(param_bounds[7, 0], param_bounds[7, 1], spacings[7]):
                                        geom_params = [h1, h2, h3, h4, r1, r2, r3, r4]
                                        geom_strs = [str(el) for el in geom_params]
                                        gfile.write(",".join(geom_strs) + '\n')
    finish = time.time()
    print('total time taken = {}'.format(finish-start))

def import_data(data_dir, grid_dir, batch_size=100):
    """

    :param data_dir:
    :param grid_dir:
    :return: returns a dataset which can yield all the input data
    """

    # define input and output files
    data_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".csv")]
    outFile = os.path.join(grid_dir, 'gridDataOut.csv')

    # pull data into python, should be either for training set or eval set
    print(data_paths)

    def get_geom(data_paths):
        for file_name in data_paths:
            with open(file_name, 'r') as file:
                for line in file:
                    geom = line.split(",")[2:10]
                    # print(geom, np.shape(geom))
                    yield geom

    ds = tf.data.Dataset.from_generator(lambda: get_geom(data_paths), (tf.float32),
                                        (tf.TensorShape([8]))
                                        )
    # shuffle then split into training and validation sets
    ds = ds.batch(batch_size, drop_remainder=True)

    iterator = ds.make_one_shot_iterator()
    features = iterator.get_next()
    pred_init_op = iterator.make_initializer(ds)

    return features, pred_init_op

def main(data_dir, grid_dir, model_name, batch_size):
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'models', model_name)
    fc_filters, tconv_Fnums, tconv_dims, tconv_filters, n_filter, n_branch, \
    reg_scale = network_helper.get_parameters(ckpt_dir)

    # initialize data reader
    if len(tconv_dims) == 0:
        output_size = fc_filters[-1]
    else:
        output_size = tconv_dims[-1]
    print('defining input data')
    features, pred_init_op = import_data(data_dir=data_dir, grid_dir=grid_dir,
                                         batch_size=batch_size)

    print('making network')
    # make network
    ntwk = network_maker.CnnNetwork(features, [], utils.my_model_fn_tens, batch_size,
                                    fc_filters=fc_filters, tconv_Fnums=tconv_Fnums, tconv_dims=tconv_dims,
                                    n_filter=n_filter, n_branch=n_branch, reg_scale=reg_scale,
                                    tconv_filters=tconv_filters, make_folder=False)

    print('defining save file')
    # evaluate the results if the results do not exist or user force to re-run evaluation
    save_file = os.path.join('.', grid_dir, 'test_pred_{}.csv'.format(model_name))
    print('executing the model ...')
    pred_file, feat_file = ntwk.predict(pred_init_op, ckpt_dir=ckpt_dir, model_name=model_name, save_file=save_file)
    return pred_file, feat_file

if __name__=="__main__":
    gen_data(
        os.path.join('.', 'dataGrid'), param_bounds=np.array([[42, 52.2], [42, 52.2], [42, 52.2], [42, 52.2],
                                                             [42, 52.2], [42, 52.2], [42, 52.2], [42, 52.2]]),
        spacings=[.8,.8,.8,.8,.8,.8,.8,.8])
    modelNum = '20190218_182224'
    #import_data(os.path.join('.', 'dataIn', 'eval'), os.path.join('.', 'dataGrid'), batch_size=10, shuffle_size=100)
    #main(os.path.join('.', 'dataIn', 'eval'), os.path.join('.', 'dataGrid'), modelNum,  batch_size=1000)
