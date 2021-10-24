# formulate loss function based on supplied ground truth and outputs from network

import importlib
import tensorflow as tf
import numpy as np
import os
from model import (
    reproj_layer,
    lambSH_layer,
)
from model import vgg16 as VGG


def loss_formulate(
    albedos,
    shadow,
    nm_pred,
    lightings,
    nm_gt,
    inputs,
    dms,
    cams,
    scale_xs,
    scale_ys,
    masks,
    reproj_inputs,
    reprojInput_mask,
    pair_label,
    sup_flag,
    illu_prior_loss,
    reg_loss_flag=True,
):

    # define perceptual loss by vgg16
    vgg_path = "../vgg16.npy"
    vgg16 = VGG.Vgg16(vgg_path)

    # pre-process inputs based on gamma
    gamma = tf.constant(1.0)  # gamma is 4d constant

    albedos = rescale_2_zero_one(albedos)
    shadow = rescale_2_zero_one(shadow)
    inputs = rescale_2_zero_one(inputs)
    reproj_inputs = rescale_2_zero_one(reproj_inputs)

    sdFree_inputs = tf.pow(tf.nn.relu(tf.pow(inputs, gamma) / shadow) + 1e-4, 1 / gamma)

    # selete normal map used in rendering - gt or pred
    normals = tf.where(sup_flag, nm_gt, nm_pred)

    ###### cProj rendering loss
    # repeating elements from original batch for a number of times that is same with the number of paired images inside the sub-batch, such that realise the parallel albedos reproj computation
    reproj_tb = tf.to_float(tf.equal(pair_label, tf.transpose(pair_label)))
    reproj_tb = tf.cast(
        tf.matrix_set_diag(reproj_tb, tf.zeros([tf.shape(inputs)[0]])), tf.bool
    )
    reproj_list = tf.where(reproj_tb)
    img1_inds = tf.expand_dims(reproj_list[:, 0], axis=-1)
    img2_inds = tf.expand_dims(reproj_list[:, 1], axis=-1)
    albedo1 = tf.gather_nd(albedos, img1_inds)
    dms1 = tf.gather_nd(dms, img1_inds)
    cams1 = tf.gather_nd(cams, img1_inds)
    albedo2 = tf.gather_nd(albedos, img2_inds)
    cams2 = tf.gather_nd(cams, img2_inds)
    scale_xs1 = tf.gather_nd(scale_xs, img1_inds)
    scale_xs2 = tf.gather_nd(scale_xs, img2_inds)
    scale_ys1 = tf.gather_nd(scale_ys, img1_inds)
    scale_ys2 = tf.gather_nd(scale_ys, img2_inds)

    lightings2 = tf.gather_nd(lightings, img2_inds)
    normals1 = tf.gather_nd(normals, img1_inds)
    shadow2 = tf.gather_nd(shadow, img2_inds)

    # rotate lighting predictions
    cams_rot = tf.matmul(
        tf.reshape(cams1[:, 4:13], (-1, 3, 3)),
        tf.transpose(tf.reshape(cams2[:, 4:13], (-1, 3, 3)), (0, 2, 1)),
    )

    thetaX, thetaY, thetaZ = rotm2eul(cams_rot)
    rot = Rotation(thetaX, thetaY, thetaZ)

    # rotate SHL from source cam_coord to target cam_coord
    reproj_lightings = tf.reduce_sum(
        rot[:, :, :, tf.newaxis] * lightings2[:, tf.newaxis, :, :], axis=-2
    )

    # scale albedo map based on max and min values such that albedo values are in range (0,1)
    reproj_shadow1, reproj_mask = reproj_layer.map_reproj(
        dms1, shadow2, cams1, cams2, scale_xs1, scale_xs2, scale_ys1, scale_ys2
    )
    reproj_shadow1 = tf.clip_by_value(reproj_shadow1, 1e-4, 0.9999)

    gamma2 = 1.0
    reproj_sdFree_inputs = tf.pow(
        tf.clip_by_value(tf.pow(reproj_inputs, gamma2) / reproj_shadow1, 1e-4, 0.9999),
        1 / gamma2,
    )

    sdFree_shadings, renderings_mask = lambSH_layer.lambSH_layer(
        tf.ones_like(albedos), normals, lightings, tf.ones_like(shadow), 1.0
    )
    reproj_sdFree_shadings, _ = lambSH_layer.lambSH_layer(
        tf.ones_like(albedo1),
        normals1,
        reproj_lightings,
        tf.ones_like(reproj_shadow1),
        1.0,
    )

    reproj_albedo1, reproj_mask = reproj_layer.map_reproj(
        dms1, albedo2, cams1, cams2, scale_xs1, scale_xs2, scale_ys1, scale_ys2
    )

    reproj_albedo1 = reproj_albedo1 + tf.constant(1e-4)  # numerical stable constant

    ### scale intensities for each image
    albedo1_pixels = tf.boolean_mask(albedo1, reproj_mask)
    reproj_albedo1_pixels = tf.boolean_mask(reproj_albedo1, reproj_mask)
    reproj_err = 0.5 * tf.losses.mean_squared_error(
        cvtLab(albedo1_pixels), cvtLab(reproj_albedo1_pixels)
    )
    reproj_err += 2.5 * perceptualLoss_formulate(
        vgg16,
        albedo1,
        reproj_albedo1,
        tf.to_float(reproj_mask[:, :, :, tf.newaxis]),
    )

    sdFree_recons = tf.pow(tf.nn.relu(albedos * sdFree_shadings), 1 / gamma)
    sdFree_inputs_pixels = cvtLab(tf.boolean_mask(sdFree_inputs, renderings_mask))
    sdFree_recons_pixels = cvtLab(tf.boolean_mask(sdFree_recons, renderings_mask))
    render_err = 0.5 * tf.losses.mean_squared_error(
        sdFree_inputs_pixels, sdFree_recons_pixels
    )
    render_err += 2.5 * perceptualLoss_formulate(
        vgg16,
        sdFree_inputs,
        sdFree_recons,
        tf.to_float(renderings_mask[:, :, :, tf.newaxis]),
    )

    ## scale intensities for each image
    reproj_sdFree_renderings = tf.pow(
        tf.nn.relu(reproj_sdFree_shadings * albedo1), 1 / gamma2
    )

    reprojInput_mask = tf.cast(reprojInput_mask[:, :, :, 0], tf.bool)
    sdFree_inputs_pixels = cvtLab(
        tf.boolean_mask(reproj_sdFree_inputs, reprojInput_mask)
    )
    reproj_sdFree_renderings_pixels = cvtLab(
        tf.boolean_mask(reproj_sdFree_renderings, reprojInput_mask)
    )
    cross_render_err = 0.5 * tf.losses.mean_squared_error(
        sdFree_inputs_pixels, reproj_sdFree_renderings_pixels
    )
    cross_render_err += 2.5 * perceptualLoss_formulate(
        vgg16,
        reproj_sdFree_inputs,
        reproj_sdFree_renderings,
        tf.to_float(reprojInput_mask[:, :, :, tf.newaxis]),
    )

    ### regualarisation loss
    reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    ### scale-invariant loss

    ### compute nm_pred error
    nmSup_mask = tf.not_equal(tf.reduce_sum(nm_gt, axis=-1), 0)
    nm_gt_pixel = tf.boolean_mask(nm_gt, nmSup_mask)
    nm_pred_pixel = tf.boolean_mask(nm_pred, nmSup_mask)
    nm_prod = tf.reduce_sum(nm_pred_pixel * nm_gt_pixel, axis=-1, keepdims=True)
    nm_cosValue = tf.constant(0.9999)
    nm_prod = tf.clip_by_value(nm_prod, -nm_cosValue, nm_cosValue)
    nm_angle = tf.acos(nm_prod) + tf.constant(1e-4)
    nm_loss = tf.reduce_mean(nm_angle ** 2)

    ### compute gradient loss
    Gx = tf.constant(1 / 2) * tf.expand_dims(
        tf.expand_dims(tf.constant([[-1, 1]], dtype=tf.float32), axis=-1), axis=-1
    )
    Gy = tf.constant(1 / 2) * tf.expand_dims(
        tf.expand_dims(tf.constant([[-1], [1]], dtype=tf.float32), axis=-1), axis=-1
    )
    nm_pred_Gx = conv2d_nosum(nm_pred, Gx)
    nm_pred_Gy = conv2d_nosum(nm_pred, Gy)
    nm_pred_Gxy = tf.concat([nm_pred_Gx, nm_pred_Gy], axis=-1)
    normals_Gx = conv2d_nosum(nm_gt, Gx)
    normals_Gy = conv2d_nosum(nm_gt, Gy)
    normals_Gxy = tf.concat([normals_Gx, normals_Gy], axis=-1)
    nm_pred_smt_error = tf.losses.mean_squared_error(nm_pred_Gxy, normals_Gxy)

    ### total loss
    render_err *= tf.constant(0.1)
    reproj_err *= tf.constant(0.1)
    cross_render_err *= tf.constant(0.1)
    illu_prior_loss *= tf.constant(0.005)
    nm_pred_smt_error *= tf.constant(1.0)
    nm_loss *= tf.constant(1.0)

    if reg_loss_flag == True:
        loss = (
            render_err
            + reproj_err
            + cross_render_err
            + reg_loss
            + illu_prior_loss
            + nm_pred_smt_error
            + nm_loss
        )
    else:
        loss = (
            render_err
            + reproj_err
            + cross_render_err
            + illu_prior_loss
            + nm_pred_smt_error
            + nm_loss
        )

    return (
        loss,
        render_err,
        reproj_err,
        cross_render_err,
        reg_loss,
        illu_prior_loss,
        nm_pred_smt_error,
        nm_loss,
        sdFree_inputs,
        sdFree_shadings,
        sdFree_recons,
    )


def perceptualLoss_formulate(vgg16, renderings, inputs, masks, w_act=0.1):
    vgg_layers = ["conv1_2"]  # conv1 through conv5
    vgg_layer_weights = [1.0]

    renderings_acts = vgg16.get_vgg_activations(renderings, vgg_layers)
    refs_acts = vgg16.get_vgg_activations(inputs, vgg_layers)

    loss = 0
    masks_shape = [(200, 200), (100, 100)]

    tmp_reproj_mask = masks
    for mask_shape, w, act1, act2 in zip(
        masks_shape, vgg_layer_weights, renderings_acts, refs_acts
    ):
        act1 *= tmp_reproj_mask
        act2 *= tmp_reproj_mask
        tmp_reproj_mask_weights = tf.tile(
            tf.clip_by_value(tmp_reproj_mask, 1e-4, 0.9999), (1, 1, 1, act1.shape[-1])
        )
        loss += (
            w
            * tf.reduce_sum(tmp_reproj_mask_weights * tf.square(w_act * (act1 - act2)))
            / tf.reduce_sum(tmp_reproj_mask_weights)
        )

        tmp_reproj_mask = 1.0 - tf.nn.max_pool(
            1.0 - tmp_reproj_mask,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
        )

    loss *= 0.0005

    return loss


# input RGB is 2d tensor with shape (n_pix, 3)
def cvtLab(RGB):

    # threshold definition
    T = tf.constant(0.008856)

    # matrix for converting RGB to LUV color space
    cvt_XYZ = tf.constant(
        [
            [0.412453, 0.35758, 0.180423],
            [0.212671, 0.71516, 0.072169],
            [0.019334, 0.119193, 0.950227],
        ]
    )

    # convert RGB to XYZ
    XYZ = tf.matmul(RGB, tf.transpose(cvt_XYZ))

    # normalise for D65 white point
    XYZ /= tf.constant([[0.950456, 1.0, 1.088754]]) * 100

    mask = tf.to_float(tf.greater(XYZ, T))

    fXYZ = XYZ ** (1 / 3) * mask + (1.0 - mask) * (
        tf.constant(7.787) * XYZ + tf.constant(0.137931)
    )

    M_cvtLab = tf.constant(
        [[0.0, 116.0, 0.0], [500.0, -500.0, 0.0], [0.0, 200.0, -200.0]]
    )

    Lab = tf.matmul(fXYZ, tf.transpose(M_cvtLab)) + tf.constant([[-16.0, 0.0, 0.0]])
    mask = tf.to_float(tf.equal(Lab, tf.constant(0.0)))

    Lab += mask * tf.constant(1e-4)

    return Lab


# compute regular 2d convolution on 3d data
def conv2d_nosum(input, kernel):
    input_x = input[:, :, :, 0:1]
    input_y = input[:, :, :, 1:2]
    input_z = input[:, :, :, 2:3]

    output_x = tf.nn.conv2d(input_x, kernel, strides=(1, 1, 1, 1), padding="SAME")
    output_y = tf.nn.conv2d(input_y, kernel, strides=(1, 1, 1, 1), padding="SAME")
    output_z = tf.nn.conv2d(input_z, kernel, strides=(1, 1, 1, 1), padding="SAME")

    return tf.concat([output_x, output_y, output_z], axis=-1)


def rescale_2_zero_one(imgs):
    return imgs / 2.0 + 0.5


def Rotation(thetaX, thetaY, thetaZ):
    num_rots = tf.shape(thetaX)[0]

    # rows_x = [0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 8]
    # cols_x = [0, 2, 1, 3, 7, 5, 6, 8, 4, 6, 8]
    idx_x = [
        [0, 0],
        [1, 2],
        [2, 1],
        [3, 3],
        [4, 7],
        [5, 5],
        [6, 6],
        [6, 8],
        [7, 4],
        [8, 6],
        [8, 8],
    ]

    data_x_p90 = [
        1,
        -1,
        1,
        1,
        -1,
        -1,
        -1 / 2,
        -np.sqrt(3) / 2,
        1,
        -np.sqrt(3) / 2,
        1 / 2,
    ]
    data_x_n90 = [
        1,
        1,
        -1,
        1,
        1,
        -1,
        -1 / 2,
        -np.sqrt(3) / 2,
        -1,
        -np.sqrt(3) / 2,
        1 / 2,
    ]

    Rot_X_p90 = tf.sparse_to_dense(
        sparse_indices=idx_x, sparse_values=data_x_p90, output_shape=(9, 9)
    )
    Rot_X_n90 = tf.sparse_to_dense(
        sparse_indices=idx_x, sparse_values=data_x_n90, output_shape=(9, 9)
    )

    Rot_X_p90 = tf.tile(Rot_X_p90[tf.newaxis], (num_rots, 1, 1))
    Rot_X_n90 = tf.tile(Rot_X_n90[tf.newaxis], (num_rots, 1, 1))

    Rot_z = rot_z(thetaZ, num_rots)

    Rot_y = rot_y(thetaY, Rot_X_p90, Rot_X_n90, num_rots)

    Rot_x = rot_x(thetaX, Rot_X_p90, Rot_X_n90, num_rots)
    # return Rot_x

    Rot = tf.matmul(Rot_z, tf.matmul(Rot_y, Rot_x))

    return Rot


def rot_z(thetaZ, num_rots):
    # rows_z = [0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8]
    # cols_z = [0, 1, 3, 2, 1, 3, 4, 8, 5, 7, 6, 5, 7, 4, 8]
    idx_z = tf.constant(
        [
            [0, 0],
            [1, 1],
            [1, 3],
            [2, 2],
            [3, 1],
            [3, 3],
            [4, 4],
            [4, 8],
            [5, 5],
            [5, 7],
            [6, 6],
            [7, 5],
            [7, 7],
            [8, 4],
            [8, 8],
        ]
    )
    idx_id = tf.reshape(
        tf.tile(tf.range(num_rots)[:, tf.newaxis], (1, tf.shape(idx_z)[0])), (-1, 1)
    )
    idx_z = tf.tile(idx_z, (num_rots, 1))
    idx_z = tf.concat([idx_id, idx_z], axis=-1)

    data_Z = tf.stack(
        [
            tf.ones_like(thetaZ),
            tf.cos(thetaZ),
            tf.sin(thetaZ),
            tf.ones_like(thetaZ),
            -tf.sin(thetaZ),
            tf.cos(thetaZ),
            tf.cos(2 * thetaZ),
            tf.sin(2 * thetaZ),
            tf.cos(thetaZ),
            tf.sin(thetaZ),
            tf.ones_like(thetaZ),
            -tf.sin(thetaZ),
            tf.cos(thetaZ),
            -tf.sin(2 * thetaZ),
            tf.cos(2 * thetaZ),
        ],
        axis=-1,
    )
    data_Z = tf.reshape(data_Z, (-1,))

    return tf.sparse_to_dense(
        sparse_indices=idx_z, sparse_values=data_Z, output_shape=(num_rots, 9, 9)
    )


def rot_y(thetaY, Rot_X_p90, Rot_X_n90, num_rots):
    # rows_z = [0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8]
    # cols_z = [0, 1, 3, 2, 1, 3, 4, 8, 5, 7, 6, 5, 7, 4, 8]
    idx_z = tf.constant(
        [
            [0, 0],
            [1, 1],
            [1, 3],
            [2, 2],
            [3, 1],
            [3, 3],
            [4, 4],
            [4, 8],
            [5, 5],
            [5, 7],
            [6, 6],
            [7, 5],
            [7, 7],
            [8, 4],
            [8, 8],
        ]
    )
    idx_id = tf.reshape(
        tf.tile(tf.range(num_rots)[:, tf.newaxis], (1, tf.shape(idx_z)[0])), (-1, 1)
    )
    idx_z = tf.tile(idx_z, (num_rots, 1))
    idx_z = tf.concat([idx_id, idx_z], axis=-1)

    data_Y = tf.stack(
        [
            tf.ones_like(thetaY),
            tf.cos(thetaY),
            tf.sin(thetaY),
            tf.ones_like(thetaY),
            -tf.sin(thetaY),
            tf.cos(thetaY),
            tf.cos(2 * thetaY),
            tf.sin(2 * thetaY),
            tf.cos(thetaY),
            tf.sin(thetaY),
            tf.ones_like(thetaY),
            -tf.sin(thetaY),
            tf.cos(thetaY),
            -tf.sin(2 * thetaY),
            tf.cos(2 * thetaY),
        ],
        axis=-1,
    )
    data_Y = tf.reshape(data_Y, (-1,))

    Rot_y = tf.sparse_to_dense(
        sparse_indices=idx_z, sparse_values=data_Y, output_shape=(num_rots, 9, 9)
    )

    return tf.matmul(Rot_X_n90, tf.matmul(Rot_y, Rot_X_p90))


def rot_x(thetaX, Rot_X_p90, Rot_X_n90, num_rots):
    # rows_z = [0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8]
    # cols_z = [0, 1, 3, 2, 1, 3, 4, 8, 5, 7, 6, 5, 7, 4, 8]
    idx_z = tf.constant(
        [
            [0, 0],
            [1, 1],
            [1, 3],
            [2, 2],
            [3, 1],
            [3, 3],
            [4, 4],
            [4, 8],
            [5, 5],
            [5, 7],
            [6, 6],
            [7, 5],
            [7, 7],
            [8, 4],
            [8, 8],
        ]
    )
    idx_id = tf.reshape(
        tf.tile(tf.range(num_rots)[:, tf.newaxis], (1, tf.shape(idx_z)[0])), (-1, 1)
    )
    idx_z = tf.tile(idx_z, (num_rots, 1))
    idx_z = tf.concat([idx_id, idx_z], axis=-1)

    data_X = tf.stack(
        [
            tf.ones_like(thetaX),
            tf.cos(thetaX),
            tf.sin(thetaX),
            tf.ones_like(thetaX),
            -tf.sin(thetaX),
            tf.cos(thetaX),
            tf.cos(2 * thetaX),
            tf.sin(2 * thetaX),
            tf.cos(thetaX),
            tf.sin(thetaX),
            tf.ones_like(thetaX),
            -tf.sin(thetaX),
            tf.cos(thetaX),
            -tf.sin(2 * thetaX),
            tf.cos(2 * thetaX),
        ],
        axis=-1,
    )
    data_X = tf.reshape(data_X, (-1,))

    Rot_x = tf.sparse_to_dense(
        sparse_indices=idx_z, sparse_values=data_X, output_shape=(num_rots, 9, 9)
    )

    half_pi = tf.tile(tf.constant([np.pi / 2]), (num_rots,))
    Rot_Y_n90 = rot_y(-half_pi, Rot_X_p90, Rot_X_n90, num_rots)
    Rot_Y_p90 = rot_y(half_pi, Rot_X_p90, Rot_X_n90, num_rots)

    return tf.matmul(Rot_Y_p90, tf.matmul(Rot_x, Rot_Y_n90))


def rotm2eul(rotm):
    sy = tf.sqrt(rotm[:, 0, 0] ** 2 + rotm[:, 1, 0] ** 2)
    singular = sy < 1e-6

    thetaX = tf.where(
        singular,
        tf.atan2(-rotm[:, 1, 2], rotm[:, 1, 1]),
        tf.atan2(rotm[:, 2, 1], rotm[:, 2, 2]),
    )

    # thetaY = tf.where(singular, tf.atan2(-rotm[:,2,0], sy), tf.atan2(rotm[:,2,1], rotm[:,2,2]))
    thetaY = tf.atan2(rotm[:, 2, 0], sy)

    thetaZ = tf.where(
        singular, tf.zeros_like(rotm[:, 0, 0]), tf.atan2(rotm[:, 1, 0], rotm[:, 0, 0])
    )

    return thetaX, thetaY, thetaZ
