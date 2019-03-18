import os
import tensorflow as tf
import numpy as np
import pandas as pd
import time

import utils
import network_maker
import network_helper
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from itertools import islice
import struct

# generate geometric parameters for the grid and save them in a file
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
                check_time = time.time()
                print('time elapsed: {}'.format(np.round(check_time-start), 1))
                print('h1 = {}, h2 = {}'.format(h1, h2))
                for h3 in np.arange(param_bounds[2, 0], param_bounds[2, 1], spacings[2]):
                    for h4 in np.arange(param_bounds[3, 0], param_bounds[3, 1], spacings[3]):
                        for r1 in np.arange(param_bounds[4, 0], param_bounds[4, 1], spacings[4]):
                            for r2 in np.arange(param_bounds[5, 0], param_bounds[5, 1], spacings[5]):
                                for r3 in np.arange(param_bounds[6, 0], param_bounds[6, 1], spacings[6]):
                                    for r4 in np.arange(param_bounds[7, 0], param_bounds[7, 1], spacings[7]):
                                        geom_params = np.round([h1, h2, h3, h4,
                                                                r1, r2, r3, r4,
                                                                r1/h1, r2/h1, r3/h1, r4/h1,
                                                                r1/h2, r2/h2, r3/h2, r4/h2,
                                                                r1/h3, r2/h3, r3/h3, r4/h3,
                                                                r1/h4, r2/h4, r3/h4, r4/h4], 1)
                                        geom_strs = [str(el) for el in geom_params]
                                        gfile.write(",".join(geom_strs) + '\n')
    finish = time.time()
    print('total time taken = {}'.format(finish-start))

# yield the geometry from the saved grid data file in the form of a dataset
def import_data(data_dir, batch_size=100):
    """
    :param data_dir:
    :param grid_dir:
    :return: returns a dataset which can yield all the input data
    """

    # define input and output files
    data_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".csv")]

    # pull data into python, should be either for training set or eval set
    print(data_paths)

    def get_geom(data_paths):
        for file_name in data_paths:
            print('getting geom from file {}'.format(file_name))
            with open(file_name, 'r') as file:
                for line in file:
                    geom = line.split(",")  # [2:26] if using validation set for testing
                    # print(geom, np.shape(geom))
                    assert len(geom) == 8 + 16, "expected geometry vector of length 8+16, got length {}".format(len(geom))
                    yield geom

    ds = tf.data.Dataset.from_generator(lambda: get_geom(data_paths), (tf.float32),
                                        (tf.TensorShape([24]))
                                        )
    # shuffle then split into training and validation sets
    ds = ds.batch(batch_size, drop_remainder=True)

    iterator = ds.make_one_shot_iterator()
    features = iterator.get_next()
    pred_init_op = iterator.make_initializer(ds)

    return features, pred_init_op


# generate predictions with the given model and save them to a spectrum library file
def main(data_dir, lib_dir, model_name, batch_size=10):
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'models', model_name)
    clip, fc_filters, tconv_Fnums, tconv_dims, tconv_filters, n_filter, n_branch, \
    reg_scale = network_helper.get_parameters(ckpt_dir)

    print('defining input data')
    features, pred_init_op = import_data(data_dir=data_dir,
                                         batch_size=batch_size)

    print('making network')
    # make network
    ntwk = network_maker.CnnNetwork(features, [], utils.my_model_fn_tens, batch_size, clip=clip,
                                    fc_filters=fc_filters, tconv_Fnums=tconv_Fnums, tconv_dims=tconv_dims,
                                    n_filter=n_filter, n_branch=n_branch, reg_scale=reg_scale,
                                    tconv_filters=tconv_filters, make_folder=False)

    print('defining save file')
    save_file = os.path.join('.', lib_dir)

    # evaluate the model for each geometry in the grid file
    print('executing the model ...')
    pred_file = ntwk.predictBin3(pred_init_op, ckpt_dir=ckpt_dir, model_name=model_name, save_file=save_file)
    return pred_file

def lookup(sstar, library_path, candidate_num):
    candidates = []
    start = time.time()
    # extract the defined points of sstar
    sstar_keyPoints = []
    for cnt, value in enumerate(sstar):
        if value is not None:
            sstar_keyPoints.append([cnt, value])

    with open(library_path) as lib:
        line_batch = islice(lib, 100)
        for line in line_batch:
            # line_start = time.time()
            # if cnt != 0 and (cnt % 1000) == 0:
            #     print('line is {}, time taken is {}'.format(cnt, np.round(time.time()-start, 3)))

            # get spectrum from library file
            spectrum = line.split(',')
            spectrum = [float(string) for string in spectrum]
            assert len(spectrum) == 300

            # calculate mse with desired spectrum
            errors = []

            for index, value in sstar_keyPoints:
                errors.append((spectrum[index] - value) ** 2)
            mse = np.mean(errors)

            if len(candidates) < candidate_num:  # then we need more candidates, so append
                candidates.append([spectrum, mse])
            else:  # see if this spectrum is better than any of the current candidates
                for candidate in candidates:
                    if candidate[1] > mse:
                        candidates.append([spectrum, mse])
                        candidates.sort(key=lambda x: x[1])
                        candidates = candidates[:candidate_num]  # take only the candidates with the lowest error
                        break

    print('total search time taken is {}'.format(np.round(time.time() - start, 4)))
    #convert to arrays so we can slice
    sstar_keyPoints = np.array(sstar_keyPoints)
    candidates = np.array(candidates)
    # plot the defined sstar points along with the candidate
    plt.scatter(sstar_keyPoints[:, 0],
                sstar_keyPoints[:, 1])
    for candidate in candidates[:, 0]:
        plt.plot(candidate)
    plt.show()
    return candidates

def lookupBin(sstar, library_path, geometries_path, candidate_num):
    candidates = []
    start = time.time()

    sstar_keyPoints = []
    for starcnt, value in enumerate(sstar):  # extract the defined points of sstar
        if value is not None:
            sstar_keyPoints.append((starcnt, value))

    # make generator for bytes from a file to be read
    def byte_yield(f, byte_num):
        dat = 'x'
        while dat:
            dat = f.read(byte_num)
            if dat:
                yield dat
            else:
                break
    batch_cnt = 0
    spec_cnt = 0
    simult_spectra = 2  # the number of spectra to read from the file at a time
    with open(library_path, 'rb') as lib:
        structobj = struct.Struct('B'*(300*simult_spectra))
        for byte_set in byte_yield(lib, byte_num=300*simult_spectra):  # needs exact length of a spectrum
            spectrum_batch = structobj.unpack(byte_set)  # unpack a single unsigned char to [0, 255]
            batch_cnt += 1

            # yield a single spectrum from the batch
            def spec_chunk(l, n):
                for i in range(0, len(l), n):
                    yield l[i: i + n]

            # consider each spectrum in the batch one at a time
            for spectrum in spec_chunk(spectrum_batch, 300):
                spec_cnt += 1
                # convert back to floats on [0, 1]
                assert len(spectrum) == 300

                # calculate mse with desired spectrum
                errors = []
                for index, value in sstar_keyPoints:
                    errors.append((spectrum[index]-value)**2)
                mse = np.mean(errors)

                if len(candidates) < candidate_num:  # then we need more candidates, so append
                    candidates.append([spectrum, mse, spec_cnt])
                else:  # see if this spectrum is better than any of the current candidates
                    for candidate in candidates:
                        if candidate[1] > mse:
                            candidates.append([spectrum, mse, spec_cnt])
                            candidates.sort(key=lambda x: x[1])
                            candidates = candidates[:candidate_num]  # take only the candidates with the lowest error
                            break

    print('total search time taken is {}'.format(np.round(time.time() - start, 4)))
    #convert to arrays so we can slice
    sstar_keyPoints = np.array(sstar_keyPoints)
    candidates = np.array(candidates)

    # get the geometric values from the file of features
    spec_indices = candidates[:, 2]
    print('spec_indices are {}'.format(spec_indices))
    geom_strings = []
    geom_cnt = 0
    with open(geometries_path, 'r') as geom_file:
        for line in geom_file:
            geom_cnt += 1
            if geom_cnt in spec_indices:
                geom_strings.append(line)
                if len(geom_strings) == len(candidates):
                    break
    geom_strings_split = [geometries.split(',') for geometries in geom_strings]
    geoms = []
    for geom_set in geom_strings_split:
        geoms.append([float(string) for string in geom_set])

    # rearrange geom elements so that they match the order of candidates (sorted by MSE)
    indices=sorted(range(len(candidates)), key=lambda k: candidates[k, 2])
    geoms = [geoms[i] for i in indices]
    print('geom_cnt is {}'.format(geom_cnt))
    print('geometries are {}'.format(np.array(geoms)))

    # plot the defined sstar points along with the candidate
    plt.scatter(sstar_keyPoints[:, 0],
                sstar_keyPoints[:, 1])
    for candidate in candidates[:, 0]:
        plt.plot(candidate)
    plt.show()
    return candidates, geoms

if __name__=="__main__":
    # gen_data(
    #     os.path.join('.', 'dataGrid', 'gridFiles'), param_bounds=np.array([
    #                                                          [42, 52.2], [42, 52.2], [42, 52.2], [42, 52.2],
    #                                                          [42, 52.2], [42, 52.2], [42, 52.2], [42, 52.2]]),
    #     spacings=[.8,.8,.8,.8,.8,.8,.8,.8])
    modelNum = '20190311_183831'
    # import_data(os.path.join('.', 'dataIn', 'eval'), os.path.join('.', 'dataGrid'), batch_size=100, shuffle_size=100)

    # test library computation
    # main_start = time.time()
    # main(data_dir=os.path.join('.', 'dataIn/eval'),
    #      grid_dir='dataGrid',
    #      model_name=modelNum,
    #      batch_size=1000)
    # print('main test time is {}'.format(time.time()-main_start))

    # for main library computation
    main(data_dir=os.path.join('.', 'dataGrid', 'gridFiles'),
         lib_dir=os.path.join('D:/dlmData/library20190311_183831'),
         model_name=modelNum, batch_size=20000)

    # define test sstar, see ML\lookupTest\findTestSpectra.nb
    # spec = [None for i in range(300)]
    # spec[50] = int(.7*255)
    # spec[140] = int(.2*255)
    # spec[250] = int(.1*255)
    # cand = lookupBin(sstar=spec,
    #                  library_path=os.path.join('.', 'dataGrid', 'test_pred_' + modelNum),
    #                  geometries_path=os.path.join('.', 'dataGrid', 'test_feat_{}'.format(modelNum) + '.csv'),
    #                  candidate_num=3)

    # # test of usigned integer spectra
    # practice_file = os.path.join('.', 'dataIn', 'orig', 'bp5_OutMod.csv')
    # with open(practice_file, 'r') as f:
    #     prac_data = pd.read_csv(f, header=None)
    # specs = prac_data.iloc[::, 10:]
    # specs2 = specs
    # print(len(specs2.iloc[:, 0]))
    # print(len(specs2.iloc[0]))
    # specs2 = specs2*255
    # specs2 = specs2.astype(np.uint64)
    # fig = plt.figure(figsize=(16, 6))
    # ax = fig.add_subplot(2, 2, 1)
    # ax.plot(specs2.iloc[0, :])
    # ax.plot(specs.iloc[0, :]*255)
    #
    # ax = fig.add_subplot(2, 2, 2)
    # ax.plot(specs2.iloc[1, :])
    # ax.plot(specs.iloc[1, :]*255)

    print('done.')
