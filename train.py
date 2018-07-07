import argparse
import tensorflow as tf
import utils
import data_reader
import network_maker
import network_helper


INPUT_SIZE = 2
FC_FILTERS = (50, 100, 500, 50)
TCONV_DIMS = (50, 150, 300)
TCONV_FILTERS = (16, 8, 4)
X_RANGE = [0, 1]
Y_RANGE = [i for i in range(2, 1003)]
CROSS_VAL = 5
VAL_FOLD = 0
BATCH_SIZE = 20
SHUFFLE_SIZE = 1
VERB_STEP = 25
EVAL_STEP = 250
TRAIN_STEP = 6000
LEARN_RATE = 1e-4
DECAY_STEP = 4000
DECAY_RATE = 0.5
TRAIN_FILE = 'TrainDataV9.txt'
VALID_FILE = 'TestDataV9.txt'


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
    parser.add_argument('--train-file', default=TRAIN_FILE, type=str, help='name of the training file')
    parser.add_argument('--valid-file', default=VALID_FILE, type=str, help='name of the validation file')

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
    features, labels, train_init_op, valid_init_op = reader.get_data_holder_and_init_op(
        (flags.train_file, flags.valid_file))

    # make network
    ntwk = network_maker.CnnNetwork(features, labels, utils.my_model_fn, flags.batch_size,
                                    fc_filters=flags.fc_filters, tconv_dims=flags.tconv_dims,
                                    tconv_filters=flags.tconv_filters, learn_rate=flags.learn_rate,
                                    decay_step=flags.decay_step, decay_rate=flags.decay_rate)
    # define hooks for monitoring training
    train_hook = network_helper.TrainValueHook(flags.verb_step, ntwk.loss,
                                               ckpt_dir=ntwk.ckpt_dir, write_summary=True)
    lr_hook = network_helper.TrainValueHook(flags.verb_step, ntwk.learn_rate, ckpt_dir=ntwk.ckpt_dir,
                                            write_summary=True, value_name='learning_rate')
    valid_hook = network_helper.ValidationHook(flags.eval_step, valid_init_op, ntwk.labels, ntwk.logits, ntwk.loss,
                                               ckpt_dir=ntwk.ckpt_dir, write_summary=True)
    # train the network
    ntwk.train(train_init_op, flags.train_step, [train_hook, valid_hook, lr_hook], write_summary=True)


if __name__ == '__main__':
        flags = read_flag()
        tf.reset_default_graph()
        main(flags)
