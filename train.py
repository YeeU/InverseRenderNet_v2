# also predict shadow mask and error mask

# no rotation


#### compute albedo reproj loss only on reprojection available area; compute reconstruction and its loss only based on defined area


import os
import shutil
import time

import numpy as np
import tensorflow as tf

from model import dataloader
import argparse

parser = argparse.ArgumentParser(description="InverseRenderNet++ training")
parser.add_argument(
    "--mode",
    type=str,
    default="scratch",
    choices=["scratch", "trained"],
    help="training mode",
)

parser.add_argument("--root-dir", type=str, default=None, help="Path to image data")
parser.add_argument("--batch-size", type=int, default=None, help="Training batchsize")
parser.add_argument(
    "--num-test-sc", type=int, default=1, help="Split for esting scenes"
)
parser.add_argument("--num-gpus", type=int, default=1, help="Number of available gpus")
parser.add_argument(
    "--use-GT-nm", action="store_true", help="Train with true normal map"
)
args = parser.parse_args()


def main():

    # training batches are list of numpy arrays each of which is paired data
    num_subbatch_input = args.batch_size
    dir = args.root_dir
    training_mode = args.mode
    num_test_sc = args.num_test_sc
    num_gpus = args.num_gpus
    supTrain = args.use_GT_nm

    inputs_shape = (5, 200, 200, 3)

    (
        md_next_element,
        md_trainData_init_op,
        md_testData_init_op,
        num_train_batches,
        num_test_batches,
    ) = dataloader.megaDepth_dataPipeline(
        num_subbatch_input, dir, training_mode, num_test_sc
    )

    # use image batch shape to create placeholder
    md_inputs_var = tf.reshape(
        md_next_element[0], (-1, inputs_shape[1], inputs_shape[2], inputs_shape[3])
    )
    md_dms_var = tf.reshape(md_next_element[1], (-1, inputs_shape[1], inputs_shape[2]))
    md_nms_var = tf.reshape(
        md_next_element[2], (-1, inputs_shape[1], inputs_shape[2], 3)
    )
    md_cams_var = tf.reshape(md_next_element[3], (-1, 16))
    md_scaleXs_var = tf.reshape(md_next_element[4], (-1,))
    md_scaleYs_var = tf.reshape(md_next_element[5], (-1,))
    md_masks_var = tf.reshape(
        md_next_element[6], (-1, inputs_shape[1], inputs_shape[2])
    )
    md_reproj_inputs_var = tf.reshape(
        md_next_element[7], (-1, inputs_shape[1], inputs_shape[2], inputs_shape[3])
    )
    md_reproj_mask_var = tf.reshape(
        md_next_element[8], (-1, inputs_shape[1], inputs_shape[2])
    )

    train_flag = tf.placeholder(tf.bool, ())
    supTrain_flag = tf.placeholder(tf.bool, ())

    pair_label_var = tf.constant(
        np.repeat(np.arange(num_subbatch_input), inputs_shape[0])[:, None],
        dtype=tf.float32,
    )

    # feed-foward neural network from input images to lighting and albedo
    (
        loss,
        render_err,
        reproj_err,
        cross_render_err,
        reg_loss,
        illu_prior_loss,
        nm_smt_loss,
        nm_loss,
        albedos,
        nm_pred,
        shadow,
        sdFree_inputs,
        sdFree_shadings,
        sdFree_recons,
    ) = make_parallel(
        num_gpus,
        md_inputs_var,
        md_dms_var,
        md_nms_var,
        md_cams_var,
        md_scaleXs_var,
        md_scaleYs_var,
        md_masks_var,
        md_reproj_inputs_var,
        md_reproj_mask_var,
        pair_label_var,
        train_flag,
        supTrain_flag,
        inputs_shape,
    )

    ### regualarisation loss
    reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # defined traning loop
    iters = 500
    num_subbatch = num_subbatch_input
    num_iters = np.int32(np.ceil(num_train_batches / num_subbatch))
    num_test_iters = np.int32(np.ceil(num_test_batches / num_subbatch))

    # define variable list for each of training
    g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="inverserendernet")

    # training op
    global_step = tf.Variable(1, name="global_step", trainable=False)

    g_optimizer = tf.train.AdamOptimizer(0.0005)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_step
        g_train_step = g_optimizer.minimize(
            loss + reg_loss,
            global_step=global_step,
            var_list=g_vars,
            colocate_gradients_with_ops=True,
        )

    # define saver for saving and restoring
    saver = tf.train.Saver(g_vars + [global_step])

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()

    if training_mode == "scratch":
        pass

    elif training_mode == "trained":
        saver.restore(sess, "model/model.ckpt")

    elif training_mode == "debug":
        saver.restore(sess, "model/model.ckpt")

    # save summeries
    render_err_summary = tf.summary.scalar("self_sup/render_err", render_err)
    reproj_err_summary = tf.summary.scalar("self_sup/reproj_err", reproj_err)
    cross_render_err_summary = tf.summary.scalar(
        "self_sup/cross_render_err", cross_render_err
    )
    illu_prior_loss_summary = tf.summary.scalar(
        "self_sup/illu_prior_loss", illu_prior_loss
    )
    nm_loss_summary = tf.summary.scalar("self_sup/nm_loss", nm_loss)
    nm_smt_loss_summary = tf.summary.scalar("self_sup/nm_smt_loss", nm_smt_loss)

    ori_summary = tf.summary.image("ori_img", md_inputs_var, max_outputs=15)
    am_summary = tf.summary.image("am", albedos, max_outputs=15)
    nm_summary = tf.summary.image("nm", nm_pred, max_outputs=15)
    shadow_summary = tf.summary.image("shadow", shadow, max_outputs=15)
    sdFree_shadings_summary = tf.summary.image(
        "sdFree_shadings", sdFree_shadings, max_outputs=15
    )
    sdFree_inputs_summary = tf.summary.image(
        "sdFree_inputs", sdFree_inputs, max_outputs=15
    )
    sdFree_recons_summary = tf.summary.image(
        "sdFree_recons", sdFree_recons, max_outputs=15
    )

    performance_summary = tf.summary.merge(
        [
            render_err_summary,
            reproj_err_summary,
            cross_render_err_summary,
            illu_prior_loss_summary,
            nm_loss_summary,
            nm_smt_loss_summary,
        ]
    )
    imgs_summary = tf.summary.merge(
        [
            ori_summary,
            am_summary,
            nm_summary,
            shadow_summary,
            sdFree_shadings_summary,
            sdFree_inputs_summary,
            sdFree_recons_summary,
        ]
    )

    if not (os.path.exists("summaries")):
        os.mkdir("summaries")
    summ_first = os.path.join("summaries", "first")
    if not (os.path.exists(summ_first)):
        os.mkdir(summ_first)
    else:
        shutil.rmtree(summ_first, ignore_errors=True)
    summ_writer = tf.summary.FileWriter(summ_first, sess.graph)

    # supTrain = True -> train albedo net by given nm_gt
    # supTrain = False -> train albedo net using nm_preds
    md_trainData_init_op.run()

    best_score = 100
    best_result = 0
    for i in range(1, iters + 1):
        g_loss_avg = 0
        f = open("cost.txt", "a")
        if training_mode == "trained" or training_mode == "scratch":
            for j in range(1, num_iters + 1):

                print("iter %d/%d loop %d/%d" % (i, iters, j, num_iters))
                f.write("iter %d/%d loop %d/%d" % (i, iters, j, num_iters))
                start_time_g = time.time()
                if j % 50 == 1:
                    [
                        global_step_val,
                        imgs_summary_val,
                        performance_summary_val,
                        _,
                        loss_val,
                        reg_loss_val,
                        render_err_val,
                        reproj_err_val,
                        cross_render_err_val,
                        illu_prior_val,
                        nm_smt_loss_val,
                        nm_loss_val,
                    ] = sess.run(
                        [
                            global_step,
                            imgs_summary,
                            performance_summary,
                            g_train_step,
                            loss,
                            reg_loss,
                            render_err,
                            reproj_err,
                            cross_render_err,
                            illu_prior_loss,
                            nm_smt_loss,
                            nm_loss,
                        ],
                        feed_dict={train_flag: True, supTrain_flag: supTrain},
                    )
                    summ_writer.add_summary(performance_summary_val, global_step_val)
                    summ_writer.add_summary(imgs_summary_val, global_step_val)

                else:
                    [
                        _,
                        loss_val,
                        reg_loss_val,
                        render_err_val,
                        reproj_err_val,
                        cross_render_err_val,
                        illu_prior_val,
                        nm_smt_loss_val,
                        nm_loss_val,
                    ] = sess.run(
                        [
                            g_train_step,
                            loss,
                            reg_loss,
                            render_err,
                            reproj_err,
                            cross_render_err,
                            illu_prior_loss,
                            nm_smt_loss,
                            nm_loss,
                        ],
                        feed_dict={train_flag: True, supTrain_flag: supTrain},
                    )

                g_loss_avg += loss_val

                if j % 1 == 0:
                    print(
                        "\tg_loss_avg = %f, loss = %f, took %.3fs"
                        % (g_loss_avg / j, loss_val, time.time() - start_time_g)
                    )
                    print(
                        "\t\treg_loss = %f, render_err = %f, reproj_err = %f, cross_render_err = %f, illu_prior = %f, nm_smt_loss = %f, nm_loss = %f"
                        % (
                            reg_loss_val,
                            render_err_val,
                            reproj_err_val,
                            cross_render_err_val,
                            illu_prior_val,
                            nm_smt_loss_val,
                            nm_loss_val,
                        )
                    )

                    f.write(
                        "\tg_loss_avg = %f, loss = %f, took %.3fs\n\t\treg_loss = %f, render_err = %f, reproj_err = %f, cross_render_err = %f, illu_prior = %f, nm_smt_loss = %f, nm_loss = %f\n"
                        % (
                            g_loss_avg / j,
                            loss_val,
                            time.time() - start_time_g,
                            reg_loss_val,
                            render_err_val,
                            reproj_err_val,
                            cross_render_err_val,
                            illu_prior_val,
                            nm_smt_loss_val,
                            nm_loss_val,
                        )
                    )

            f.close()

            md_testData_init_op.run()
            test_loss = 0
            test_render_err = 0
            test_reproj_err = 0
            test_cross_render_err = 0
            test_illu_prior = 0
            test_nm_loss = 0
            for j in range(1, num_test_iters + 1):
                [
                    loss_val,
                    reg_loss_val,
                    render_err_val,
                    reproj_err_val,
                    cross_render_err_val,
                    illu_prior_val,
                    nm_smt_loss_val,
                    nm_loss_val,
                ] = sess.run(
                    [
                        loss,
                        reg_loss,
                        render_err,
                        reproj_err,
                        cross_render_err,
                        illu_prior_loss,
                        nm_smt_loss,
                        nm_loss,
                    ],
                    feed_dict={train_flag: False, supTrain_flag: supTrain},
                )

                test_loss += loss_val
                test_render_err += render_err_val
                test_reproj_err += reproj_err_val
                test_cross_render_err += cross_render_err_val
                test_illu_prior += illu_prior_val
                test_nm_loss += nm_loss_val

            test_loss /= num_test_iters
            test_render_err /= num_test_iters
            test_reproj_err /= num_test_iters
            test_cross_render_err /= num_test_iters
            test_illu_prior /= num_test_iters
            test_nm_loss /= num_test_iters

            score = test_loss

            if best_score > score:
                best_result = i
                best_score = score
                saver.save(sess, "model_best/model.ckpt")

            f = open("test.txt", "a")
            f.write(
                "iter {:d}, score {:f}: render_err={:f}, reproj_err={:f}, cross_render_err={:f}, illu_prior={:f}, nm_loss={:f}\n".format(
                    i,
                    score,
                    test_render_err,
                    test_reproj_err,
                    test_cross_render_err,
                    test_illu_prior,
                    test_nm_loss,
                )
            )
            f.write(
                "\tbest_result {:d}, best_score {:f}\n".format(best_result, best_score)
            )
            f.close()

            md_trainData_init_op.run()

            # save model every 10 iterations
            if i % 1 == 0:
                saver.save(sess, "model/model.ckpt")


def make_parallel(
    num_gpus,
    inputs_var,
    dms_var,
    nms_var,
    cams_var,
    scaleXs_var,
    scaleYs_var,
    masks_var,
    reproj_inputs_var,
    reproj_mask_var,
    pair_label_var,
    train_flag,
    supTrain_flag,
    inputs_shape,
):
    from model import SfMNet, consistency_layer

    inputs_var = tf.split(inputs_var, num_gpus)
    dms_var = tf.split(dms_var, num_gpus)
    nms_var = tf.split(nms_var, num_gpus)
    cams_var = tf.split(cams_var, num_gpus)
    scaleXs_var = tf.split(scaleXs_var, num_gpus)
    scaleYs_var = tf.split(scaleYs_var, num_gpus)
    masks_var = tf.split(masks_var, num_gpus)
    reproj_inputs_var = tf.split(reproj_inputs_var, num_gpus)
    reproj_mask_var = tf.split(reproj_mask_var, num_gpus)
    pair_label_var = tf.split(pair_label_var, num_gpus)

    loss_split = []
    render_err_split = []
    reproj_err_split = []
    cross_render_err_split = []
    reg_loss_split = []
    illu_prior_loss_split = []
    nm_smt_loss_split = []
    nm_loss_split = []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                # mask out sky in inputs and nms
                # dms_var *= masks_var
                masks_var_4d = tf.expand_dims(masks_var[i], axis=-1)
                reproj_mask_var_4d = tf.expand_dims(reproj_mask_var[i], axis=-1)

                inputs_var[i] *= masks_var_4d
                nms_var[i] *= masks_var_4d

                albedos, shadow, nm_pred = SfMNet.SfMNet(
                    inputs=inputs_var[i],
                    is_training=train_flag,
                    height=inputs_shape[1],
                    width=inputs_shape[2],
                    masks=masks_var_4d,
                    n_layers=30,
                    n_pools=4,
                    depth_base=32,
                )

                normals = tf.where(supTrain_flag, nms_var[i], nm_pred)

                # linearise srgb input to rgb
                rbg_inputs_var = inputs_srbg_2_rbg(inputs_var[i])
                rbg_reproj_inputs_var = inputs_srbg_2_rbg(reproj_inputs_var[i])

                # infer lighting from rgb input and compute lighting loss
                lightings, illu_prior_loss = SfMNet.comp_light(
                    rbg_inputs_var, albedos, normals, shadow, 1.0, masks_var_4d
                )

                (
                    loss,
                    render_err,
                    reproj_err,
                    cross_render_err,
                    reg_loss,
                    illu_prior_loss,
                    nm_smt_loss,
                    nm_loss,
                    sdFree_inputs,
                    sdFree_shadings,
                    sdFree_recons,
                ) = consistency_layer.loss_formulate(
                    albedos,
                    shadow,
                    nm_pred,
                    lightings,
                    nms_var[i],
                    rbg_inputs_var,
                    dms_var[i],
                    cams_var[i],
                    scaleXs_var[i],
                    scaleYs_var[i],
                    masks_var_4d,
                    rbg_reproj_inputs_var,
                    reproj_mask_var_4d,
                    pair_label_var[i],
                    supTrain_flag,
                    illu_prior_loss,
                    reg_loss_flag=False,
                )

                loss_split += [loss]
                render_err_split += [render_err]
                reproj_err_split += [reproj_err]
                cross_render_err_split += [cross_render_err]
                reg_loss_split += [reg_loss]
                illu_prior_loss_split += [illu_prior_loss]
                nm_smt_loss_split += [nm_smt_loss]
                nm_loss_split += [nm_loss]

    loss = tf.reduce_mean(loss_split)
    render_err = tf.reduce_mean(render_err_split)
    reproj_err = tf.reduce_mean(reproj_err_split)
    cross_render_err = tf.reduce_mean(cross_render_err_split)
    reg_loss = tf.reduce_mean(reg_loss_split)
    illu_prior_loss = tf.reduce_mean(illu_prior_loss_split)
    nm_smt_loss = tf.reduce_mean(nm_smt_loss_split)
    nm_loss = tf.reduce_mean(nm_loss_split)

    return (
        loss,
        render_err,
        reproj_err,
        cross_render_err,
        reg_loss,
        illu_prior_loss,
        nm_smt_loss,
        nm_loss,
        albedos,
        nm_pred,
        shadow,
        sdFree_inputs,
        sdFree_shadings,
        sdFree_recons,
    )


def inputs_srbg_2_rbg(imgs):
    imgs = imgs / 2.0 + 0.5

    ret = tf.zeros_like(imgs)
    dp_mask = tf.to_float(imgs <= 0.04045)
    ret += dp_mask * imgs / 12.92
    ret += tf.pow((imgs + 0.055) / 1.055, 2.2) * (1 - dp_mask)
    imgs = tf.identity(ret)

    imgs = imgs * 2.0 - 1.0

    return imgs


if __name__ == "__main__":
    main()
