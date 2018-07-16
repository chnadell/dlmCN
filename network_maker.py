import os
import time
import inspect
import numpy as np
import tensorflow as tf
import utils


class CnnNetwork(object):
    def __init__(self, features, labels, model_fn, batch_size, fc_filters=(5, 10, 15),
                 tconv_dims=(60, 120, 240), tconv_filters=(1, 1, 1),
                 learn_rate=1e-4, decay_step=200, decay_rate=0.1,
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
        return self.model_fn(self.features, self.batch_size, self.fc_filters, self.tconv_dims, self.tconv_filters)

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


class BiganNetwork(CnnNetwork):
    def __init__(self, features, labels, model_fn, batch_size, fc_filters=(5, 10, 15),
                 tconv_dims=(60, 120, 240), tconv_filters=(1, 1, 1),
                 learn_rate=1e-4, decay_step=200, decay_rate=0.1,
                 ckpt_dir=os.path.join(os.path.dirname(__file__), 'models'),
                 make_folder=True, class_num=2):
        self.class_num = class_num
        super(BiganNetwork, self).__init__(features, labels, model_fn, batch_size, fc_filters, tconv_dims,
                                           tconv_filters, learn_rate, decay_step, decay_rate, ckpt_dir, make_folder)



    def create_graph(self):
        """
        Create model graph
        :return: outputs of the last layer
        """
        fc = self.features
        lb = self.labels

        # encoder
        with tf.variable_scope('encoder'):
            for cnt, filters in enumerate(self.fc_filters):
                fc = tf.layers.dense(inputs=fc, units=filters, activation=tf.nn.leaky_relu, name='fc_up{}'.format(cnt),
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            encode_fc = fc

            down = lb
            down = tf.expand_dims(down, axis=2)
            for cnt, down_filter in enumerate(self.tconv_filters[::-1]):
                down = tf.layers.conv1d(down, down_filter, 3, activation=tf.nn.leaky_relu, name='down{}'.format(cnt),
                                        padding='same')
                down = tf.layers.max_pooling1d(down, 2, 2)
            encode_lb = tf.layers.conv1d(down, 1, 1, activation=None, name='down_final', padding='same')
            encode_lb = tf.squeeze(encode_lb, axis=2)

        # generator
        with tf.variable_scope('generator'):
            up = tf.expand_dims(encode_fc, axis=2)
            feature_dim = self.fc_filters[-1]
            last_filter = 1
            for cnt, (up_size, up_filter) in enumerate(zip(self.tconv_dims, self.tconv_filters)):
                assert up_size % feature_dim == 0
                stride = up_size // feature_dim
                feature_dim = up_size
                f = tf.Variable(tf.random_normal([3, up_filter, last_filter]))
                up = utils.conv1d_transpose(up, f, [self.batch_size, up_size, up_filter], stride, name='up{}'.format(cnt))
                last_filter = up_filter
            gener = tf.squeeze(tf.layers.conv1d(up, 1, 1, activation=None, name='conv_final'), axis=2)

        # descriminator
        with tf.variable_scope('discriminator'):
            fc = encode_lb
            for cnt, filters in enumerate(self.fc_filters[::-1]):
                fc = tf.layers.dense(inputs=fc, units=filters, activation=tf.nn.leaky_relu, name='fc_down{}'.format(cnt),
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            discr = tf.layers.dense(inputs=fc, units=self.class_num, activation=None, name='fc_fianl',
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        return encode_fc, encode_lb, gener, discr

    def make_loss(self):
        """
        Make cross entropy loss
        :return: mean cross entropy loss of the batch
        """
        with tf.variable_scope('loss'):
            g_loss = tf.losses.mean_squared_error(self.labels, self.logits[2])
            d_loss = tf.losses.mean_squared_error(self.features, self.logits[3])
            e_loss = tf.losses.mean_squared_error(self.logits[0], self.logits[1])
            return [g_loss, d_loss, e_loss]

    def make_optimizer(self):
        """
        Make an Adam optimizer with the learning rate defined when the class is initialized
        :return: an AdamOptimizer
        """
        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if 'generator' in var.name]
        e_vars = [var for var in t_vars if 'encoder' in var.name]
        d_vars = [var for var in t_vars if 'discriminator' in var.name]

        g_optm = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.loss[0], self.global_step,
                                                                                var_list=g_vars)
        d_optm = tf.train.AdamOptimizer(learning_rate=self.learn_rate / 2).minimize(self.loss[1], self.global_step,
                                                                                    var_list=d_vars)
        e_optm = tf.train.AdamOptimizer(learning_rate=self.learn_rate / 2).minimize(self.loss[2], self.global_step,
                                                                                    var_list=e_vars)

        return [g_optm, d_optm, e_optm]

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
                sess.run(self.optm[0])
                sess.run(self.optm[0])
                sess.run(self.optm[1])
                sess.run(self.optm[2])

                for hook in hooks:
                    hook.run(sess, writer=summary_writer)
            self.save(sess)


class AutoEncoderNetwork(CnnNetwork):
    def __init__(self, features, labels, model_fn, batch_size, fc_filters=(5, 10, 15),
                 tconv_dims=(60, 120, 240), tconv_filters=(1, 1, 1),
                 encode_conv_filters=(1, 2, 4), encode_fc_filters=(50, 15, 10),
                 learn_rate=1e-4, decay_step=200, decay_rate=0.1,
                 ckpt_dir=os.path.join(os.path.dirname(__file__), 'models'),
                 make_folder=True, class_num=2):
        self.class_num = class_num
        self.encode_conv_filters = encode_conv_filters
        self.encode_fc_filters = encode_fc_filters
        super(AutoEncoderNetwork, self).__init__(features, labels, model_fn, batch_size, fc_filters, tconv_dims,
                                                 tconv_filters, learn_rate, decay_step, decay_rate, ckpt_dir,
                                                 make_folder)

    def create_graph(self):
        """
        Create model graph
        :return: outputs of the last layer
        """
        fc = self.features
        lb = self.labels

        def make_encoder(lb, reuse=False):
            with tf.variable_scope('encoder', reuse=reuse):
                down = lb
                down = tf.expand_dims(down, axis=2)
                for cnt, down_filter in enumerate(self.encode_conv_filters):
                    down = tf.layers.conv1d(down, down_filter, 3, activation=tf.nn.leaky_relu,
                                            name='down{}'.format(cnt),
                                            padding='same')
                    down = tf.layers.max_pooling1d(down, 2, 2)
                down_dim = self.tconv_dims[-1] // (2 ** len(self.encode_conv_filters))
                fc = tf.reshape(down, [self.batch_size, down_dim * self.encode_conv_filters[-1]], name='down_flat')
                for cnt, filters in enumerate(self.encode_fc_filters):
                    fc = tf.layers.dense(inputs=fc, units=filters, activation=tf.nn.leaky_relu,
                                         name='fc_down{}'.format(cnt),
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))
                discr = tf.layers.dense(inputs=fc, units=self.class_num, activation=None, name='fc_fianl',
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.02))
                return discr

        def make_decoder(fc, reuse=False):
            with tf.variable_scope('decoder', reuse=reuse):
                for cnt, filters in enumerate(self.fc_filters):
                    fc = tf.layers.dense(inputs=fc, units=filters, activation=tf.nn.leaky_relu,
                                         name='fc_up{}'.format(cnt),
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))
                encode_fc = fc
                up = tf.expand_dims(encode_fc, axis=2)
                feature_dim = self.fc_filters[-1]
                last_filter = 1
                for cnt, (up_size, up_filter) in enumerate(zip(self.tconv_dims, self.tconv_filters)):
                    assert up_size % feature_dim == 0
                    stride = up_size // feature_dim
                    feature_dim = up_size
                    f = tf.Variable(tf.random_normal([3, up_filter, last_filter]))
                    up = utils.conv1d_transpose(up, f, [self.batch_size, up_size, up_filter], stride,
                                                name='up{}'.format(cnt))
                    last_filter = up_filter
                gener = tf.squeeze(tf.layers.conv1d(up, 1, 1, activation=None, name='conv_final'), axis=2)
                return gener

        # encoder
        encode = make_encoder(lb, reuse=False)
        decode_real = make_decoder(fc, reuse=False)
        decode_fake = make_decoder(encode, reuse=True)

        return encode, decode_real, decode_fake

    def make_loss(self):
        """
        Make cross entropy loss
        :return: mean cross entropy loss of the batch
        """
        with tf.variable_scope('loss'):
            e_loss = tf.losses.mean_squared_error(self.features, self.logits[0])
            g_loss_real = tf.losses.mean_squared_error(self.labels, self.logits[1])
            g_loss_fake = tf.losses.mean_squared_error(self.labels, self.logits[2])
            return [e_loss, g_loss_real, g_loss_fake]

    def make_optimizer(self):
        """
        Make an Adam optimizer with the learning rate defined when the class is initialized
        :return: an AdamOptimizer
        """
        t_vars = tf.trainable_variables()
        e_vars = [var for var in t_vars if 'encoder' in var.name]
        g_vars_real = [var for var in t_vars if 'decoder' in var.name]
        g_vars_fake = [var for var in t_vars if 'encoder' in var.name] + \
                      [var for var in t_vars if 'decoder' in var.name]

        e_optm = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.loss[0], self.global_step,
                                                                                var_list=e_vars)
        g_real_optm = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.loss[1], self.global_step,
                                                                                     var_list=g_vars_real)
        g_fake_optm = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.loss[2], self.global_step,
                                                                                     var_list=g_vars_fake)

        return [e_optm, g_real_optm, g_fake_optm]

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
                sess.run(self.optm[0])
                sess.run(self.optm[1])
                sess.run(self.optm[2])

                for hook in hooks:
                    hook.run(sess, writer=summary_writer)
            self.save(sess)

