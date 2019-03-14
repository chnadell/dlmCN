import argparse
import tensorflow as tf
import utils
import data_reader
import network_maker
import network_helper

INPUT_SIZE = 2
CLIP = 15
FC_FILTERS = (100, 500, 1000, 1500, 500, 2000, 1000, 500, 165)
TCONV_FNUMS = (4, 4, 4)
TCONV_DIMS = (165, 165, 330)
TCONV_FILTERS = (8, 4, 4)
N_FILTER = [15]
N_BRANCH = 2
REG_SCALE = 5e-8
CROSS_VAL = 5
VAL_FOLD = 0
BATCH_SIZE = 10
SHUFFLE_SIZE = 2000
VERB_STEP = 25
EVAL_STEP = 500
TRAIN_STEP = 45000
LEARN_RATE = 1e-4
DECAY_STEP = 20000
DECAY_RATE = 0.05
X_RANGE = [i for i in range(2, 10 + 16)]
Y_RANGE = [i for i in range(10 + 16, 2011 + 16)]
# TRAIN_FILE = 'bp2_OutMod.csv'
# VALID_FILE = 'bp2_OutMod.csv'


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', type=int, default=INPUT_SIZE, help='input size')
    parser.add_argument('--clip', type=int, default=CLIP, help='points clipped from each end of output after final conv')
    parser.add_argument('--fc-filters', type=tuple, default=FC_FILTERS, help='#neurons in each fully connected layers')
    parser.add_argument('--tconv-Fnums', type=tuple, default=TCONV_FNUMS, help='#0th shape dim of each tconv layer')
    parser.add_argument('--tconv-dims', type=tuple, default=TCONV_DIMS,
                        help='dimensionality of data after each transpose convolution')
    parser.add_argument('--tconv-filters', type=tuple, default=TCONV_FILTERS,
                        help='#filters at each transpose convolution')
    parser.add_argument('--n-filter', type=int, default=N_FILTER, help='#neurons in the tensor module'),
    parser.add_argument('--n-branch', type=int, default=N_BRANCH, help='#parallel branches in the tensor module')
    parser.add_argument('--reg-scale', type=float, default=REG_SCALE, help='#scale for regularization of dense layers')
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
    # parser.add_argument('--train-file', default=TRAIN_FILE, type=str, help='name of the training file')
    # parser.add_argument('--valid-file', default=VALID_FILE, type=str, help='name of the validation file')

    flags = parser.parse_args()
    return flags

def main(flags):
    # initialize data reader
    if len(flags.tconv_dims) == 0:
        output_size = flags.fc_filters[-1]
    else:
        output_size = flags.tconv_dims[-1]

    features, labels, train_init_op, valid_init_op = data_reader.read_data(input_size=flags.input_size,
                                                                           output_size=output_size-2*flags.clip,
                                                                           x_range=flags.x_range,
                                                                           y_range=flags.y_range,
                                                                           cross_val=flags.cross_val,
                                                                           val_fold=flags.val_fold,
                                                                           batch_size=flags.batch_size,
                                                                           shuffle_size=flags.shuffle_size)

    # make network
    ntwk = network_maker.CnnNetwork(features, labels, utils.my_model_fn_tens, flags.batch_size,
                                    clip=flags.clip, fc_filters=flags.fc_filters, tconv_Fnums=flags.tconv_Fnums,
                                    tconv_dims=flags.tconv_dims,
                                    tconv_filters=flags.tconv_filters, n_filter=flags.n_filter,
                                    n_branch=flags.n_branch, reg_scale=flags.reg_scale,
                                    learn_rate=flags.learn_rate,
                                    decay_step=flags.decay_step, decay_rate=flags.decay_rate)
    # define hooks for monitoring training
    train_hook = network_helper.TrainValueHook(flags.verb_step, ntwk.loss,
                                               ckpt_dir=ntwk.ckpt_dir, write_summary=True)
    lr_hook = network_helper.TrainValueHook(flags.verb_step, ntwk.learn_rate, ckpt_dir=ntwk.ckpt_dir,
                                            write_summary=True, value_name='learning_rate')
    valid_hook = network_helper.ValidationHook(flags.eval_step, valid_init_op, ntwk.labels, ntwk.logits, ntwk.loss,
                                               ntwk.preconv, ntwk.preTconv,
                                               ckpt_dir=ntwk.ckpt_dir, write_summary=True)
    # train the network
    ntwk.train(train_init_op, flags.train_step, [train_hook, valid_hook, lr_hook], write_summary=True)


if __name__ == '__main__':
        flags = read_flag()
        tf.reset_default_graph()
        main(flags)
