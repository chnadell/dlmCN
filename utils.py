import tensorflow as tf

# Set of functions for implementing the tensor module from Ng et al.
# https://cs.stanford.edu/~danqi/papers/nips2013.pdf
# closer to implementation by Ma et al.
# https://pubs.acs.org/doi/10.1021/acsnano.8b03569

# repeats a rank 1 tensor, used to compute
# multiplication with the rank 3 kernel
def repeat_2d(input_, axis, batch_size, repeat_num):
    with tf.variable_scope('repeat_2d'):
        orig_shape = input_.get_shape().as_list()
        input_ = tf.tile(input_, (1, repeat_num, 1))
        orig_shape.insert(axis, repeat_num)
        # orig_shape[orig_shape.index(None)] = batch_size
        return tf.reshape(input_, shape=orig_shape)


def tensor_layer(input_, out_dim, batch_size, layer_id):
    # D is considered column vector here, not the row vector as in the paper
    in_dim = input_.get_shape().as_list()[1]
    var_w = tf.get_variable(name='w_k_{}'.format(layer_id), shape=[out_dim, in_dim, in_dim],
                            initializer=tf.keras.initializers.glorot_normal())
    var_w = tf.broadcast_to(var_w, [batch_size, out_dim, in_dim, in_dim])

    var_v = tf.get_variable(name='v_k_{}'.format(layer_id), shape=[out_dim, in_dim],
                            initializer=tf.keras.initializers.glorot_normal())
    var_v = tf.broadcast_to(var_v, [batch_size, out_dim, in_dim])

    var_b = tf.get_variable(name='v_b_{}'.format(layer_id), shape=[out_dim, 1])
    var_d = tf.expand_dims(input_, 1)

    # D^T * W_k * D
    temp_1 = tf.matmul(repeat_2d(var_d, 1, batch_size, out_dim), var_w)
    temp_1 = tf.matmul(temp_1, repeat_2d(tf.transpose(var_d, perm=[0, 2, 1]), 1, batch_size, out_dim))
    # V_k*D
    temp_2 = tf.matmul(var_v, tf.transpose(var_d, perm=[0, 2, 1]))

    return tf.nn.relu(temp_1[:, :, 0, 0] + temp_2[:, :, 0] + tf.broadcast_to(var_b, [batch_size, out_dim, 1])[:, :, 0])


def tensor_module(input_, out_dim, batch_size, n_filter, n_branch):
    vec_concat = []
    with tf.variable_scope('tensor_module'):
        for i in range(n_branch):
            with tf.variable_scope('tensor_layer{}'.format(i)):
                fc = tensor_layer(input_, out_dim, batch_size, i)
            for cnt, filters in enumerate(n_filter):
                with tf.variable_scope('fc{}_branch{}'.format(i, cnt)):
                    fc = tf.layers.dense(inputs=fc, units=filters, activation=None,
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            vec_concat.append(fc)
        return tf.concat(vec_concat, 1)

"""conv1d_tranpose function"""
def conv1d_transpose_wrap(value,
                          filter,
                          output_shape,
                          stride,
                          padding="SAME",
                          data_format="NWC",
                          name=None):
    """Wrap the built-in (contrib) conv1d_transpose function so that output
    has a batch size determined at runtime, rather than being fixed by whatever
    batch size was used during training"""

    dyn_input_shape = tf.shape(value)
    batch_size = dyn_input_shape[0]
    output_shape = tf.stack([batch_size, output_shape[1], output_shape[2]])

    return tf.contrib.nn.conv1d_transpose(
        value,
        filter,
        output_shape,
        stride,
        padding=padding,
        data_format=data_format,
        name=name
    )


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or 'Linear'):
        matrix = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable('bias', [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def my_model_fn(features, batch_size, fc_filters, tconv_dims, tconv_filters, n_filter, n_branch):
    """
    My customized model function
    :param features: input features
    :param output_size: dimension of output data
    :return:
    """
    fc = features
    for cnt, filters in enumerate(fc_filters):
        fc = tf.layers.dense(inputs=fc, units=filters, activation=tf.nn.leaky_relu, name='fc{}'.format(cnt),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.02))
    up = tf.expand_dims(fc, axis=2)
    feature_dim = fc_filters[-1]

    last_filter = 1
    for cnt, (up_size, up_filter) in enumerate(zip(tconv_dims, tconv_filters)):
        assert up_size%feature_dim == 0, "up_size={} while feature_dim={} (cnt={})! " \
                                        "Thus mod is {}".format(up_size, feature_dim, cnt, up_size%feature_dim)
        stride = up_size // feature_dim
        feature_dim = up_size
        f = tf.Variable(tf.random_normal([3, up_filter, last_filter]))
        up = conv1d_transpose_wrap(up, f, [batch_size, up_size, up_filter], stride, name='up{}'.format(cnt))
        last_filter = up_filter

    up = tf.layers.conv1d(up, 1, 1, activation=None, name='conv_final')

    return tf.squeeze(up, axis=2)


def my_model_fn_linear(features, batch_size, fc_filters, tconv_dims, tconv_filters):
    """
    My customized model function
    :param features: input features
    :param output_size: dimension of output data
    :return:
    """
    fc = features
    for cnt, filters in enumerate(fc_filters):
        fc = linear(fc, filters, 'fc_linear_{}'.format(cnt), with_w=False)
        fc = tf.nn.leaky_relu(fc)
    return fc


def my_model_fn_linear_conv1d(features, batch_size, fc_filters, tconv_dims, tconv_filters):
    """
    My customized model function
    :param features: input features
    :param output_size: dimension of output data
    :return:
    """
    fc = features
    for cnt, filters in enumerate(fc_filters):
        fc = linear(fc, filters, 'fc_linear_{}'.format(cnt), with_w=False)
        fc = tf.nn.leaky_relu(fc)

    up = tf.expand_dims(fc, axis=2)
    feature_dim = fc_filters[-1]

    last_filter = 1
    for cnt, (up_size, up_filter) in enumerate(zip(tconv_dims, tconv_filters)):
        assert up_size%feature_dim == 0
        stride = up_size // feature_dim
        feature_dim = up_size
        f = tf.Variable(tf.random_normal([3, up_filter, last_filter]))
        up = conv1d_transpose_wrap(up, f, [batch_size, up_size, up_filter], stride, name='up{}'.format(cnt))
        up = tf.layers.conv1d(up, up_filter, 3, activation=tf.nn.leaky_relu, name='conv_up{}'.format(cnt),
                              padding='same')
        last_filter = up_filter
    preconv = up
    up = tf.layers.conv1d(preconv, 1, 1, activation=None, name='conv_final')

    return tf.squeeze(up, axis=2), preconv

def my_model_fn_tens(features, batch_size, fc_filters, tconv_fNums, tconv_dims, tconv_filters, n_filter, n_branch, reg_scale):
    """
    My customized model function
    :param features: input features
    :param output_size: dimension of output data
    :return:
    """
    fc = features
    fc = tensor_module(fc, fc_filters[0], batch_size, n_filter, n_branch)

    for cnt, filters in enumerate(fc_filters):
        fc = tf.layers.dense(inputs=fc, units=filters, activation=tf.nn.leaky_relu, name='fc{}'.format(cnt),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))
    preTconv = fc
    up = tf.expand_dims(preTconv, axis=2)
    feature_dim = fc_filters[-1]

    last_filter = 1
    for cnt, (up_fNum, up_size, up_filter) in enumerate(zip(tconv_fNums, tconv_dims, tconv_filters)):
        assert up_size % feature_dim == 0, "up_size={} while feature_dim={} (cnt={})! " \
                                        "Thus mod is {}".format(up_size, feature_dim, cnt, up_size%feature_dim)
        stride = up_size // feature_dim
        feature_dim = up_size
        f = tf.Variable(tf.random_normal([up_fNum, up_filter, last_filter]))
        up = conv1d_transpose_wrap(up, f, [batch_size, up_size, up_filter], stride, name='up{}'.format(cnt))
        last_filter = up_filter
    preconv = up
    up = tf.layers.conv1d(preconv, 1, 1, activation=None, name='conv_final')
    up = tf.squeeze(up, axis=2)
    # up = tf.layers.dense(inputs=up, units=tconv_dims[-1], activation=tf.nn.leaky_relu, name='fc_final',
    #                           kernel_initializer=tf.random_normal_initializer(stddev=0.02))

    return up, preconv, preTconv
