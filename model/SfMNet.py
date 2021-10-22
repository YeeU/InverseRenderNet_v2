import importlib
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
import os
from model import pred_illuDecomp_layer_new as pred_illuDecomp_layer


def SfMNet(
    inputs,
    height,
    width,
    masks,
    n_layers=12,
    n_pools=2,
    is_training=True,
    depth_base=64,
):
    conv_layers = np.int32(n_layers / 2) - 1
    deconv_layers = np.int32(n_layers / 2)
    # number of layers before perform pooling
    nlayers_befPool = np.int32(np.ceil((conv_layers - 1) / n_pools) - 1)

    max_depth = 512

    # dimensional arrangement
    # number of layer at tail where no pooling anymore
    # also exclude first layer who in charge of expanding dimension
    if depth_base * 2 ** n_pools < max_depth:
        tail = conv_layers - nlayers_befPool * n_pools
        tail_deconv = deconv_layers - nlayers_befPool * n_pools
    else:
        maxNum_pool = np.log2(max_depth / depth_base)
        tail = np.int32(conv_layers - nlayers_befPool * maxNum_pool)
        tail_deconv = np.int32(deconv_layers - nlayers_befPool * maxNum_pool)

    f_in_conv = (
        [3]
        + [
            np.int32(depth_base * 2 ** (np.ceil(i / nlayers_befPool) - 1))
            for i in range(1, conv_layers - tail + 1)
        ]
        + [
            np.int32(depth_base * 2 ** maxNum_pool)
            for i in range(conv_layers - tail + 1, conv_layers + 1)
        ]
    )
    f_out_conv = (
        [64]
        + [
            np.int32(depth_base * 2 ** (np.floor(i / nlayers_befPool)))
            for i in range(1, conv_layers - tail + 1)
        ]
        + [
            np.int32(depth_base * 2 ** maxNum_pool)
            for i in range(conv_layers - tail + 1, conv_layers + 1)
        ]
    )

    f_in_deconv = f_out_conv[:0:-1] + [64]
    f_out_amDeconv = f_in_conv[:0:-1] + [3]
    f_out_MaskDeconv = f_in_conv[:0:-1] + [1]
    f_out_nmDeconv = f_in_conv[:0:-1] + [2]

    group_norm_params = {
        "groups": 16,
        "channels_axis": -1,
        "reduction_axes": (-3, -2),
        "center": True,
        "scale": True,
        "epsilon": 1e-4,
        "param_initializers": {
            "beta_initializer": tf.zeros_initializer(),
            "gamma_initializer": tf.ones_initializer(),
            "moving_variance_initializer": tf.ones_initializer(),
            "moving_average_initializer": tf.zeros_initializer(),
        },
    }

    # contractive conv_layer block
    conv_out = inputs
    conv_out_list = []
    for i, f_in, f_out in zip(range(1, conv_layers + 2), f_in_conv, f_out_conv):
        scope = "inverserendernet/conv" + str(i)

        if (
            np.mod(i - 1, nlayers_befPool) == 0
            and i <= n_pools * nlayers_befPool + 1
            and i != 1
        ):
            conv_out_list.append(conv_out)
            conv_out = conv2d(conv_out, scope, f_in, f_out)
            conv_out = tf.nn.max_pool(
                conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
            )
        else:
            conv_out = conv2d(conv_out, scope, f_in, f_out)

    # expanding deconv_layer block succeeding conv_layer block
    am_deconv_out = conv_out
    for i, f_in, f_out in zip(range(1, deconv_layers + 1), f_in_deconv, f_out_amDeconv):
        scope = "inverserendernet/am_deconv" + str(i)

        # expand resolution every after nlayers_befPool deconv_layer
        if np.mod(i, nlayers_befPool) == 0 and i <= n_pools * nlayers_befPool:
            # attach previous convolutional output to upsampling/deconvolutional output
            tmp = conv_out_list[-np.int32(i / nlayers_befPool)]
            output_shape = tmp.shape[1:3]
            am_deconv_out = tf.image.resize_images(am_deconv_out, output_shape)
            am_deconv_out = conv2d(am_deconv_out, scope, f_in, f_out)
            am_deconv_out = tf.concat([am_deconv_out, tmp], axis=-1)
        elif i == deconv_layers:
            # no normalisation and activation, which is placed at the end
            am_deconv_out = layers.conv2d(
                am_deconv_out,
                num_outputs=f_out,
                kernel_size=[3, 3],
                stride=[1, 1],
                padding="SAME",
                normalizer_fn=None,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(
                    mean=0, stddev=np.sqrt(2 / 9 / f_in)
                ),
                weights_regularizer=layers.l2_regularizer(scale=1e-5),
                scope=scope,
            )
        else:
            # layers that not expand spatial resolution
            am_deconv_out = conv2d(am_deconv_out, scope, f_in, f_out)

    # deconvolution net for nm estimates
    nm_deconv_out = conv_out
    for i, f_in, f_out in zip(range(1, deconv_layers + 1), f_in_deconv, f_out_nmDeconv):
        scope = "inverserendernet/nm" + str(i)

        # expand resolution every after nlayers_befPool deconv_layer
        if np.mod(i, nlayers_befPool) == 0 and i <= n_pools * nlayers_befPool:
            # attach previous convolutional output to upsampling/deconvolutional output
            tmp = conv_out_list[-np.int32(i / nlayers_befPool)]
            output_shape = tmp.shape[1:3]
            nm_deconv_out = tf.image.resize_images(nm_deconv_out, output_shape)
            nm_deconv_out = conv2d(nm_deconv_out, scope, f_in, f_out)
            nm_deconv_out = tf.concat([nm_deconv_out, tmp], axis=-1)
        elif i == deconv_layers:
            # no normalisation and activation, which is placed at the end
            nm_deconv_out = layers.conv2d(
                nm_deconv_out,
                num_outputs=f_out,
                kernel_size=[3, 3],
                stride=[1, 1],
                padding="SAME",
                normalizer_fn=None,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(
                    mean=0, stddev=np.sqrt(2 / 9 / f_in)
                ),
                weights_regularizer=layers.l2_regularizer(scale=1e-5),
                biases_initializer=None,
                scope=scope,
            )
        else:
            # layers that not expand spatial resolution
            nm_deconv_out = conv2d(nm_deconv_out, scope, f_in, f_out)

    # deconv branch for predicting masks
    mask_deconv_out = conv_out
    for i, f_in, f_out in zip(
        range(1, deconv_layers + 1), f_in_deconv, f_out_MaskDeconv
    ):
        scope = "inverserendernet/mask_deconv" + str(i)

        # expand resolution every after nlayers_befPool deconv_layer
        if np.mod(i, nlayers_befPool) == 0 and i <= n_pools * nlayers_befPool:
            # with tf.variable_scope(scope):
            # attach previous convolutional output to upsampling/deconvolutional output
            tmp = conv_out_list[-np.int32(i / nlayers_befPool)]
            output_shape = tmp.shape[1:3]
            mask_deconv_out = tf.image.resize_images(mask_deconv_out, output_shape)
            mask_deconv_out = conv2d(mask_deconv_out, scope, f_in, f_out)
            mask_deconv_out = tf.concat([mask_deconv_out, tmp], axis=-1)
        elif i == deconv_layers:
            # no normalisation and activation, which is placed at the end
            mask_deconv_out = layers.conv2d(
                mask_deconv_out,
                num_outputs=f_out,
                kernel_size=[3, 3],
                stride=[1, 1],
                padding="SAME",
                normalizer_fn=None,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(
                    mean=0, stddev=np.sqrt(2 / 9 / f_in)
                ),
                weights_regularizer=layers.l2_regularizer(scale=1e-5),
                scope=scope,
            )
        else:
            # layers that not expand spatial resolution
            mask_deconv_out = conv2d(mask_deconv_out, scope, f_in, f_out)

    albedos = am_deconv_out[:, :, :, :3]
    nm_pred = nm_deconv_out

    albedos = tf.clip_by_value(tf.nn.tanh(albedos) * masks, -0.9999, 0.9999)

    nm_pred_norm = tf.sqrt(
        tf.reduce_sum(nm_pred ** 2, axis=-1, keepdims=True) + tf.constant(1.0)
    )
    nm_pred_xy = nm_pred / nm_pred_norm
    nm_pred_z = tf.constant(1.0) / nm_pred_norm
    nm_pred_xyz = tf.concat([nm_pred_xy, nm_pred_z], axis=-1) * masks

    shadow = mask_deconv_out[:, :, :, :1]
    shadow = tf.clip_by_value(tf.nn.tanh(shadow) * masks, -0.9999, 0.9999)

    return albedos, shadow, nm_pred_xyz


def get_bilinear_filter(filter_shape, upscale_factor):
    ##filter_shape is [width, height, num_in_channels, num_out_channels]
    kernel_size = filter_shape[1]
    # Centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    x, y = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size))
    bilinear = (1 - abs((x - centre_location) / upscale_factor)) * (
        1 - abs((y - centre_location) / upscale_factor)
    )
    weights = np.tile(
        bilinear[:, :, None, None], (1, 1, filter_shape[2], filter_shape[3])
    )

    return tf.constant_initializer(weights)


def group_norm(inputs, scope="group_norm"):
    input_shape = tf.shape(inputs)
    _, H, W, C = inputs.get_shape().as_list()
    group = 32
    with tf.variable_scope(scope):
        gamma = tf.get_variable(
            "scale",
            shape=[C],
            dtype=tf.float32,
            initializer=tf.ones_initializer(),
            trainable=True,
            regularizer=layers.l2_regularizer(scale=1e-5),
        )

        beta = tf.get_variable(
            "bias",
            shape=[C],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=True,
            regularizer=layers.l2_regularizer(scale=1e-5),
        )

        inputs = tf.reshape(inputs, [-1, H, W, group, C // group], name="unpack")
        mean, var = tf.nn.moments(inputs, [1, 2, 4], keep_dims=True)
        inputs = (inputs - mean) / tf.sqrt(var + 1e-5)
        inputs = tf.reshape(inputs, input_shape, name="pack")
        gamma = tf.reshape(gamma, [1, 1, 1, C], name="reshape_gamma")
        beta = tf.reshape(beta, [1, 1, 1, C], name="reshape_beta")
        return inputs * gamma + beta


def conv2d(inputs, scope, f_in, f_out):
    conv_out = layers.conv2d(
        inputs,
        num_outputs=f_out,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding="SAME",
        normalizer_fn=None,
        activation_fn=None,
        weights_initializer=tf.random_normal_initializer(
            mean=0, stddev=np.sqrt(2 / 9 / f_in)
        ),
        weights_regularizer=layers.l2_regularizer(scale=1e-5),
        biases_initializer=None,
        scope=scope,
    )

    with tf.variable_scope(scope):
        gn_out = group_norm(conv_out)

        relu_out = tf.nn.relu(gn_out)

    return relu_out


def comp_light(inputs, albedos, normals, shadows, gamma, masks):
    inputs = rescale_2_zero_one(inputs)
    albedos = rescale_2_zero_one(albedos)
    shadows = rescale_2_zero_one(shadows)

    lighting_model = "../hdr_illu_pca"
    lighting_vectors = tf.constant(
        np.load(os.path.join(lighting_model, "pcaVector.npy")), dtype=tf.float32
    )
    lighting_means = tf.constant(
        np.load(os.path.join(lighting_model, "mean.npy")), dtype=tf.float32
    )
    lightings_var = tf.constant(
        np.load(os.path.join(lighting_model, "pcaVariance.npy")), dtype=tf.float32
    )

    lightings = pred_illuDecomp_layer.illuDecomp(
        inputs, albedos, normals, shadows, gamma, masks
    )
    lightings_pca = tf.matmul((lightings - lighting_means), pinv(lighting_vectors))

    # recompute lightings from lightins_pca which could add weak constraint on lighting reconstruction
    lightings = tf.matmul(lightings_pca, lighting_vectors) + lighting_means
    # reshape 27-D lightings to 9*3 lightings
    lightings = tf.reshape(lightings, [tf.shape(lightings)[0], 9, 3])

    # lighting prior loss
    var = tf.reduce_mean(lightings_pca ** 2, axis=0)

    illu_prior_loss = tf.losses.absolute_difference(var, lightings_var)
    illu_prior_loss = tf.constant(0.0)

    return lightings, illu_prior_loss


def pinv(A, reltol=1e-6):
    # compute SVD of input A
    s, u, v = tf.svd(A)

    # invert s and clear entries lower than reltol*s_max
    atol = tf.reduce_max(s) * reltol
    # s = tf.boolean_mask(s, s>atol)
    s = tf.where(s > atol, s, atol * tf.ones_like(s))
    s_inv = tf.diag(1.0 / s)
    # s_inv = tf.diag(tf.concat([1./s, tf.zeros([tf.size(b) - tf.size(s)])], axis=0))

    # compute v * s_inv * u_t as psuedo inverse
    return tf.matmul(v, tf.matmul(s_inv, tf.transpose(u)))


def rescale_2_zero_one(imgs):
    return imgs / 2.0 + 0.5


def rescale_2_minusOne_one(imgs):
    return imgs * 2.0 - 1.0
