import os
import time
import inspect
import numpy as np
import tensorflow as tf
import struct


class CnnNetwork(object):
    def __init__(self, features, labels, model_fn, batch_size, clip=0, fc_filters=(5, 10, 15),
                 tconv_Fnums=(4,4), tconv_dims=(60, 120, 240), tconv_filters=(1, 1, 1),
                 n_filter=5, n_branch=3, reg_scale=.001, learn_rate=1e-4, decay_step=200, decay_rate=0.1,
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
        self.clip = clip
        self.fc_filters = fc_filters
        assert len(tconv_dims) == len(tconv_filters)
        assert len(tconv_Fnums) == len(tconv_filters)
        self.tconv_Fnums = tconv_Fnums
        self.tconv_dims = tconv_dims
        self.tconv_filters = tconv_filters
        self.n_filter = n_filter
        self.n_branch = n_branch
        self.reg_scale = reg_scale
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
        self.learn_rate = tf.train.exponential_decay(learn_rate, self.global_step,
                                                     decay_step, decay_rate, staircase=True)

        self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.gmtime()))
        if not os.path.exists(self.ckpt_dir) and make_folder:
            os.makedirs(self.ckpt_dir)
            self.write_record()

        self.logits, self.preconv, self.preTconv = self.create_graph()
        if self.labels==[]:
            print('labels list is empty')
        else:
            self.loss = self.make_loss()
            self.optm = self.make_optimizer()

    def create_graph(self):
        """
        Create model graph
        :return: outputs of the last layer
        """
        return self.model_fn(self.features, self.batch_size, self.clip, self.fc_filters, self.tconv_Fnums,
                             self.tconv_dims, self.tconv_filters,
                             self.n_filter, self.n_branch, self.reg_scale)

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
            loss = tf.losses.mean_squared_error(self.labels, self.logits)
            #loss = tf.reduce_mean(tf.math.pow(x=tf.cast(self.labels, dtype=tf.float32)-self.logits, y=4, name ='L4')) # L4 loss
            loss += tf.losses.get_regularization_loss()
            return loss
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
            feat_file = os.path.join(save_file, 'test_feat.csv')
            with open(pred_file, 'w'), open(truth_file, 'w'):
                pass
            try:
                while True:
                    with open(pred_file, 'a') as f1, open(truth_file, 'a') as f2, open(feat_file, 'a') as f3:
                        pred, truth, features = sess.run([self.logits, self.labels, self.features])
                        np.savetxt(f1, pred, fmt='%.3f')
                        np.savetxt(f2, truth, fmt='%.3f')
                        np.savetxt(f3, features, fmt='%.3f')
            except tf.errors.OutOfRangeError:
                return pred_file, truth_file

    def predict(self, pred_init_op, ckpt_dir, save_file=os.path.join(os.path.dirname(__file__), 'dataGrid'),
                model_name=''):
        """
        Evaluate the model, and save predictions to save_file
        :param ckpt_dir directory
        :param save_file: full path to pred file
        :param model_name: name of the model
        :return:
        """
        with tf.Session() as sess:
            self.load(sess, ckpt_dir)
            sess.run(pred_init_op)
            pred_file = os.path.join(save_file, 'test_pred_{}.csv'.format(model_name))
            feat_file = os.path.join(save_file, 'test_feat.csv')
            with open(pred_file, 'w'):
                pass
            try:
                start = time.time()
                cnt = 1
                while True:
                    with open(pred_file, 'a') as f1: #, open(feat_file, 'a') as f2
                        pred_batch, features_batch = sess.run([self.logits, self.features])
                        for pred, features in zip(pred_batch, features_batch):
                            pred_str = [str(el) for el in pred]
                            features_str = [ str(el) for el in features]
                            f1.write(','.join(pred_str)+'\n')
                            # f2.write(','.join(features_str)+'\n')
                    if (cnt % 100) == 0:
                        print('cnt is {}, time elapsed is {}, features are {} '.format(cnt,
                                                                                       np.round(time.time()-start),
                                                                                       features_batch))
                    cnt += 1
            except tf.errors.OutOfRangeError:
                return pred_file, feat_file
                pass

    def predictBin(self, pred_init_op, ckpt_dir, save_file=os.path.join(os.path.dirname(__file__), 'dataGrid'),
                model_name=''):
        """
        Evaluate the model, and save predictions to binary save_file
        :param ckpt_dir directory
        :param save_file: full path to pred file
        :param model_name: name of the model
        :return:
        """
        with tf.Session() as sess:
            self.load(sess, ckpt_dir)
            sess.run(pred_init_op)
            pred_file = os.path.join(save_file, 'test_pred_{}'.format(model_name))
            feat_file = os.path.join(save_file, 'test_feat_{}'.format(model_name) + '.csv')

            try:
                start = time.time()
                cnt = 1
                while True:
                    with open(pred_file, 'ab') as f1:
                        pred_batch, features_batch = sess.run([self.logits, self.features])
                        for pred, features in zip(pred_batch, features_batch):
                            # occasionally predicts a slightly negative value, so clip these out network
                            preduint64 = [int(np.round(x*255)) for x in np.clip(pred, a_min=0, a_max=1)]
                            pred_bin = struct.pack("B"*len(preduint64), *preduint64)
                            f1.write(pred_bin)
                            # features_str = ','.join([str(ftr) for ftr in features])
                            # f2.write(features_str + '\n')
                            # cnt += 1
                            # if (cnt % 1000000) == 0:
                            #     print('cnt is {}, minutes elapsed is {}'.format(cnt, np.round(time.time()-start)/60))
            except tf.errors.OutOfRangeError:
                return pred_file, feat_file
                pass

    def predictBin2(self, pred_init_op, ckpt_dir, save_file=os.path.join(os.path.dirname(__file__), 'dataGrid'),
                model_name=''):
        """
        Evaluate the model, and save predictions to binary save_file
        :param ckpt_dir directory
        :param save_file: full path to pred file
        :param model_name: name of the model
        :return:
        """
        with tf.Session() as sess:
            self.load(sess, ckpt_dir)
            sess.run(pred_init_op)
            pred_file = os.path.join(save_file, 'test_pred_{}'.format(model_name))

            try:
                while True:
                    with open(pred_file, 'ab') as f1:
                        pred_batch = sess.run(self.logits)
                        pred_batch = pred_batch.flatten()
                        # occasionally predicts a slightly negative value, so clip these out network
                        preduint64 = [int(np.round(x*255)) for x in np.clip(pred_batch, a_min=0, a_max=1)]
                        pred_bin = struct.pack("B"*len(preduint64), *preduint64)
                        f1.write(pred_bin)
            except tf.errors.OutOfRangeError:
                return pred_file,
                pass

# write it to a number of different files which are smaller, using np.save()
    def predictBin3(self, pred_init_op, ckpt_dir, save_file=os.path.join(os.path.dirname(__file__), 'dataGrid'),
                model_name=''):
        """
        Evaluate the model, and save predictions to binary save_file
        :param ckpt_dir directory
        :param save_file: full path to pred file
        :param model_name: name of the model
        :return:
        """
        with tf.Session() as sess:
            self.load(sess, ckpt_dir)
            sess.run(pred_init_op)
            pred_file = os.path.join(save_file)
            file_prefix = 'test_pred_{}_'.format(model_name)
            try:
                file_cnt = 0
                while True:
                    pred_batch = sess.run(self.logits)
                    # network occasionally predicts value slightly outside [0,1], so clip these out
                    # then map [0,1] --> [0,255], int
                    preduint64 = np.array([np.round(x*255) for x in np.clip(pred_batch, a_min=0, a_max=1)]).astype('uint')
                    f = os.path.join(pred_file, file_prefix + str(file_cnt).zfill(4) + '.npy')
                    np.save(f, preduint64, allow_pickle=False)
                    file_cnt+=1
            except tf.errors.OutOfRangeError:
                return pred_file,
                pass