import argparse
import tensorflow as tf
import utils
import data_reader
import network_maker
import network_helper


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', type=int, default=INPUT_SIZE, help='input size')
    parser.add_argument('--fc-filters', type=tuple, default=FC_FILTERS, help='#neurons in each fully connected layers')
    parser.add_argument('--tconv-dims', type=tuple, default=TCONV_DIMS,
                        help='dimensionality of data after each transpose convolution')
    parser.add_argument('--tconv-filters', type=tuple, default=TCONV_FILTERS,
                        help='#filters at each transpose convolution')
    parser.add_argument('--x-range', type=list, default=X_RANGE, help='columns of input parameters')
    parser.add_argument('--y-range', type=list, default=Y_RANGE, help='columns of output parameters')
    parser.add_argument('--cross-val', type=int, default=CROSS_VAL, help='# cross validation folds')
    parser.add_argument('--val-fold', type=int, default=VAL_FOLD, help='fold to be used for validation')
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (100)')
    parser.add_argument('--shuffle-size', default=SHUFFLE_SIZE, type=int, help='shuffle size (100)')
    parser.add_argument('--verb-step', default=VERB_STEP, type=int, help='# steps between every print message')
    parser.add_argument('--eval-step', default=EVAL_STEP, type=int, help='# steps between evaluations')
    parser.add_argument('--train-step', default=TRAIN_STEP, type=int, help='# steps to train on the dataset')
    parser.add_argument('--learn-rate', default=LEARN_RATE, type=float, help='learning rate')
    parser.add_argument('--decay-step', default=DECAY_STEP, type=int,
                        help='decay learning rate at this number of steps')
    parser.add_argument('--decay-rate', default=DECAY_RATE, type=float,
                        help='decay learn rate by multiplying this factor')

    flags = parser.parse_args()
    return flags


def main(flags):
    # initialize data reader
    if len(flags.tconv_dims) == 0:
        output_size = flags.fc_filters[-1]
    else:
        output_size = flags.tconv_dims[-1]
    reader = data_reader.DataReader(input_size=flags.input_size, output_size=output_size,
                                    x_range=flags.x_range, y_range=flags.y_range, cross_val=flags.cross_val,
                                    val_fold=flags.val_fold, batch_size=flags.batch_size,
                                    shuffle_size=flags.shuffle_size)
    features, labels, train_init_op, valid_init_op = reader.get_data_holder_and_init_op()

    # make network
    ntwk = network_maker.CnnNetwork(features, labels, utils.my_model_fn, flags.batch_size,
                                    fc_filters=flags.fc_filters, tconv_dims=flags.tconv_dims,
                                    tconv_filters=flags.tconv_filters, learn_rate=flags.learn_rate,
                                    decay_step=flags.decay_step, decay_rate=flags.decay_rate)
    # define hooks for monitoring training
    train_hook = network_helper.TrainValueHook(flags.verb_step, ntwk.loss,
                                               ckpt_dir=ntwk.ckpt_dir, write_summary=True)
    valid_hook = network_helper.ValidationHook(flags.eval_step, valid_init_op, ntwk.labels, ntwk.logits, ntwk.loss,
                                               ntwk.preconv, ntwk.preTconv, ckpt_dir=ntwk.ckpt_dir, write_summary=True)
    # train the network
    ntwk.train(train_init_op, flags.train_step, [train_hook, valid_hook], write_summary=True)


if __name__ == '__main__':
    INPUT_SIZE = 2
    FC = [(5, 10, 15, 30), (5, 10, 20, 50), (5, 10, 15, 30), (5, 10, 20, 50), (5, 10, 15, 30), (5, 10, 20, 50)]
    TD = [(),              (),              (60, 120, 240),  (100, 200, 400), (60, 120, 240),  (100, 200, 400)]
    TF = [(),              (),              (1, 1, 1),       (1, 1, 1),       (4, 8, 16),      (4, 8, 16)]
    for FC_FILTERS, TCONV_DIMS, TCONV_FILTERS in zip(FC, TD, TF):
        #FC_FILTERS = (5, 10, 15, 30)
        #TCONV_DIMS = (60, 120, 240)
        #TCONV_FILTERS = (4, 8, 16)
        X_RANGE = [0, 1]
        Y_RANGE = [i for i in range(2, 1002)]
        CROSS_VAL = 5
        VAL_FOLD = 0
        BATCH_SIZE = 20
        SHUFFLE_SIZE = 5
        VERB_STEP = 25
        EVAL_STEP = 250
        TRAIN_STEP = 1500
        LEARN_RATE = 1e-3
        DECAY_STEP = 10000
        DECAY_RATE = 0.96

        flags = read_flag()
        tf.reset_default_graph()
        main(flags)
