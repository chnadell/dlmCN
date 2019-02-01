import os
import time
import inspect
import numpy as np
import tensorflow as tf


class CnnNetwork(object):
    def __init__(self, features, labels, model_fn, batch_size, fc_filters=(5, 10, 15),
                 tconv_dims=(60, 120, 240), tconv_filters=(1, 1, 1),
                 n_filter=5, n_branch=3, learn_rate=1e-4, decay_step=200, decay_rate=0.1,
                 ckpt_dir=os.path.join(os.path.dirname(__file__), 'models'),
                 make_folder=True):
        """
        Initialize a Network class
        :param features: input features
        :param labels: input labels
        :param model_fn: model definition function, can be customized by user
        :param batch_size: batch size
        :param fc_filters: #neurons in each fully connected layers
        :param tconv_dims: dimensionality of data after each transpose convolution
        :param tconv_filters: #filters at each transpose convolution
        :param learn_rate: learning rate
        :param decay_step: decay learning rate at this number of steps
        :param decay_rate: decay learn rate by multiplying this factor
        :param ckpt_dir: checkpoint directory, default to ./models
        :param make_folder: if True, create the directory if not exists
        """
        self.features = features
        self.labels = labels
        self.model_fn = model_fn
        self.batch_size = batch_size
        self.fc_filters = fc_filters
        assert len(tconv_dims) == len(tconv_filters)
        self.tconv_dims = tconv_dims
        self.tconv_filters = tconv_filters
        self.n_filter = n_filter
        self.n_branch = n_branch
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
        self.learn_rate = tf.train.exponential_decay(learn_rate, self.global_step,
                                                     decay_step, decay_rate, staircase=True)

        self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.gmtime()))
        if not os.path.exists(self.ckpt_dir) and make_folder:
            os.makedirs(self.ckpt_dir)
            self.write_record()

        self.logits = self.create_graph()
        self.loss = self.make_loss()
        self.optm = self.make_optimizer()

    def create_graph(self):
        """
        Create model graph
        :return: outputs of the last layer
        """
        return self.model_fn(self.features, self.batch_size, self.fc_filters, self.tconv_dims, self.tconv_filters,
                             self.n_filter, self.n_branch)

    def write_record(self):
        """
        Write records, including model_fn, parameters into the checkpoint folder
        These records can be used to reconstruct & repeat experiments
        :return:
        """
        model_fn_str = inspect.getsource(self.model_fn)
        params = inspect.getmembers(self, lambda a: not inspect.isroutine(a))
        params = [a for a in params if not (a[0].startswith('__') and a[0].endswith('__'))]
        with open(os.path.join(self.ckpt_dir, 'model_meta.txt'), 'w+') as f:
            f.write('model_fn:\n')
            f.writelines(model_fn_str)
            f.write('\nparams:\n')
            for key, val in params:
                f.write('{}: {}\n'.format(key, val))

    def make_loss(self):
        """
        Make cross entropy loss
        :return: mean cross entropy loss of the batch
        """
        with tf.variable_scope('loss'):
            return tf.losses.mean_squared_error(self.labels, self.logits)
            #return tf.reduce_mean(tf.nn.l2_loss(self.labels - self.logits))

    def make_optimizer(self):
        """
        Make an Adam optimizer with the learning rate defined when the class is initialized
        :return: an AdamOptimizer
        """
        return tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.loss, self.global_step)

    def save(self, sess):
        """
        Save the model to the checkpoint directory
        :param sess: current running session
        :return:
        """
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
        saver.save(sess, os.path.join(self.ckpt_dir, 'model.ckpt'))

    def load(self, sess, ckpt_dir):
        """
        Load the model from the checkpoint directory
        :param sess: current running session
        :param ckpt_dir: checkpoint directory
        :return:
        """
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        saver = tf.train.Saver(var_list=tf.global_variables())
        latest_check_point = tf.train.latest_checkpoint(ckpt_dir)
        saver.restore(sess, latest_check_point)
        print('loaded {}'.format(latest_check_point))

    def train(self, train_init_op, step_num, hooks, write_summary=False):
        """
        Train the model with step_num steps
        :param train_init_op: training dataset init operation
        :param step_num: number of steps to train
        :param hooks: hooks for monitoring the training process
        :param write_summary: write summary into tensorboard of not
        :return:
        """
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            if write_summary:
                summary_writer = tf.summary.FileWriter(self.ckpt_dir, sess.graph)
            else:
                summary_writer = None

            for i in range(int(step_num)):
                sess.run(train_init_op)
                sess.run(self.optm)

                for hook in hooks:
                    hook.run(sess, writer=summary_writer)
            self.save(sess)

    def evaluate(self, valid_init_op, ckpt_dir, save_file=os.path.join(os.path.dirname(__file__), 'data'),
                 model_name=''):
        """
        Evaluate the model, and save predictions to save_file
        :param valid_init_op: validation dataset init operation
        :param checkpoint directory
        :param save_file: full path to pred file
        :param model_name: name of the model
        :return:
        """
        with tf.Session() as sess:
            self.load(sess, ckpt_dir)
            sess.run(valid_init_op)
            pred_file = os.path.join(save_file, 'test_pred_{}.csv'.format(model_name))
            truth_file = os.path.join(save_file, 'test_truth.csv')
            with open(pred_file, 'w'), open(truth_file, 'w'):
                pass
            try:
                while True:
                    with open(pred_file, 'a') as f1, open(truth_file, 'a') as f2:
                        pred, truth = sess.run([self.logits, self.labels])
                        np.savetxt(f1, pred, fmt='%.2f')
                        np.savetxt(f2, truth, fmt='%.2f')
            except tf.errors.OutOfRangeError:
                return pred_file, truth_file
                pass
