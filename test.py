import ipdb
from tqdm import tqdm
import json
import os
import numpy as np
import tensorflow as tf
import shutil
import cv2
from skimage import io
from model import lambSH_layer, SfMNet
from utils.render_sphere_nm import render_sphere_nm
from utils.whdr import compute_whdr
from utils.diode_metrics import angular_error
import argparse

parser = argparse.ArgumentParser(description="InverseRenderNet++")
parser.add_argument(
    "--mode",
    type=str,
    default="demo_im",
    choices=["demo_im", "iiw", "diode"],
    help="testing mode",
)

# test demo image
parser.add_argument("--image", type=str, default=None, help="Path to test image")
parser.add_argument("--mask", type=str, default=None, help="Path to image mask")
# test iiw
parser.add_argument(
    "--iiw", type=str, default=None, help="Root directory for iiw-dataset"
)
# test diode
parser.add_argument(
    "--diode", type=str, default=None, help="Root directory for iiw-dataset"
)
# model and output path
parser.add_argument("--model", type=str, required=True, help="Path to trained model")
parser.add_argument("--output", type=str, required=True, help="Folder saving outputs")
args = parser.parse_args()


def rescale_2_zero_one(imgs):
    return imgs / 2.0 + 0.5


def srgb_to_rgb(srgb):
    """Taken from bell2014: sRGB -> RGB."""
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret


def irn_func(input_height, input_width):
    # define inputs
    inputs_var = tf.placeholder(tf.float32, (None, input_height, input_width, 3))
    masks_var = tf.placeholder(tf.float32, (None, input_height, input_width, 1))
    train_flag = tf.placeholder(tf.bool, ())

    albedos, shadow, nm_pred = SfMNet.SfMNet(
        inputs=inputs_var,
        is_training=train_flag,
        height=input_height,
        width=input_width,
        masks=masks_var,
        n_layers=30,
        n_pools=4,
        depth_base=32,
    )

    gamma = tf.constant(2.2)
    lightings, _ = SfMNet.comp_light(
        inputs_var, albedos, nm_pred, shadow, gamma, masks_var
    )

    # rescale
    albedos = rescale_2_zero_one(albedos) * masks_var
    shadow = rescale_2_zero_one(shadow)
    inputs = rescale_2_zero_one(inputs_var) * masks_var

    # visualise lighting on a sphere
    num_rendering = tf.shape(lightings)[0]
    nm_sphere = tf.constant(render_sphere_nm(100, 1), dtype=tf.float32)
    nm_sphere = tf.tile(nm_sphere, (num_rendering, 1, 1, 1))
    lighting_recon, _ = lambSH_layer.lambSH_layer(
        tf.ones_like(nm_sphere), nm_sphere, lightings, tf.ones_like(nm_sphere), 1.0
    )

    # recon shading map
    shading, _ = lambSH_layer.lambSH_layer(
        tf.ones_like(albedos), nm_pred, lightings, tf.ones_like(albedos), 1.0
    )

    return (
        albedos,
        shadow,
        nm_pred,
        lighting_recon,
        shading,
        inputs,
        inputs_var,
        masks_var,
        train_flag,
    )


def post_process(
    albedos_val,
    shading_val,
    shadow_val,
    lighting_recon_val,
    nm_pred_val,
    ori_width,
    ori_height,
    resize=True,
):
    # post-process results
    results = {}

    if resize:
        results.update(
            dict(albedos=cv2.resize(albedos_val[0], (ori_width, ori_height)))
        )

        results.update(
            dict(shading=cv2.resize(shading_val[0], (ori_width, ori_height)))
        )

        results.update(
            dict(shadow=cv2.resize(shadow_val[0, :, :, 0], (ori_width, ori_height)))
        )

        results.update(dict(lighting_recon=lighting_recon_val[0]))

        results.update(
            dict(nm_pred=cv2.resize(nm_pred_val[0], (ori_width, ori_height)))
        )
    else:
        results.update(dict(albedos=albedos_val[0]))

        results.update(dict(shading=shading_val[0]))

        results.update(dict(shadow=shadow_val[0, ..., 0]))

        results.update(dict(lighting_recon=lighting_recon_val[0]))

        results.update(dict(nm_pred=nm_pred_val[0]))

    return results


def saving_result(results, dst_dir, prefix=""):
    img = np.uint8(results["img"])
    albedos = np.uint8(results["albedos"] * 255.0)
    shading = np.uint8(results["shading"] * 255.0)
    shadow = np.uint8(results["shadow"] * 255.0)
    lighting_recon = np.uint8(results["lighting_recon"] * 255.0)
    nm_pred = np.uint8(results["nm_pred"] * 255.0)

    # save images
    input_path = os.path.join(dst_dir, prefix + "img.png")
    io.imsave(input_path, img)
    nm_pred_path = os.path.join(dst_dir, prefix + "nm_pred.png")
    io.imsave(nm_pred_path, nm_pred)
    albedo_path = os.path.join(dst_dir, prefix + "albedo.png")
    io.imsave(albedo_path, albedos)
    shading_path = os.path.join(dst_dir, prefix + "shading.png")
    io.imsave(shading_path, shading)
    shadow_path = os.path.join(dst_dir, prefix + "shadow.png")
    io.imsave(shadow_path, shadow)
    lighting_path = os.path.join(dst_dir, prefix + "lighting.png")
    io.imsave(lighting_path, lighting_recon)
    pass


def rescale_img(img):
    img_h, img_w = img.shape[:2]
    if img_h > img_w:
        scale = img_w / 200
        new_img_h = np.int32(img_h / scale)
        new_img_w = 200

        img = cv2.resize(img, (new_img_w, new_img_h))
    else:
        scale = img_h / 200
        new_img_w = np.int32(img_w / scale)
        new_img_h = 200

        img = cv2.resize(img, (new_img_w, new_img_h))

    return img, (img_h, img_w), (new_img_h, new_img_w)


if args.mode == "demo_im":
    assert args.image is not None and args.mask is not None

    # read in images
    img_path = args.image
    mask_path = args.mask

    img = io.imread(img_path)
    mask = io.imread(mask_path)

    input_height = 200
    input_width = 200

    ori_img, (ori_height, ori_width), (input_height, input_width) = rescale_img(img)

    # run inverse rendering
    (
        albedos,
        shadow,
        nm_pred,
        lighting_recon,
        shading,
        inputs,
        inputs_var,
        masks_var,
        train_flag,
    ) = irn_func(input_height, input_width)

    # load model and run session
    model_path = tf.train.get_checkpoint_state(args.model).model_checkpoint_path
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    # evaluation
    dst_dir = args.output
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir, ignore_errors=True)
    os.makedirs(dst_dir)

    imgs = np.float32(ori_img) / 255.0
    imgs = srgb_to_rgb(imgs)
    imgs = imgs * 2.0 - 1.0
    imgs = imgs[None]
    mask = cv2.resize(mask, (input_width, input_height), cv2.INTER_NEAREST)
    img_masks = np.float32(mask == 255)[None, ..., None]
    imgs *= img_masks
    [
        albedos_val,
        nm_pred_val,
        shadow_val,
        lighting_recon_val,
        shading_val,
        inputs_val,
    ] = sess.run(
        [albedos, nm_pred, shadow, lighting_recon, shading, inputs],
        feed_dict={inputs_var: imgs, masks_var: img_masks, train_flag: False},
    )

    # post-process results
    results = post_process(
        albedos_val,
        shading_val,
        shadow_val,
        lighting_recon_val,
        nm_pred_val,
        ori_width,
        ori_height,
    )

    # rescale albedo and normal
    results["albedos"] = (results["albedos"] - results["albedos"].min()) / (
        results["albedos"].max() - results["albedos"].min()
    )

    results["nm_pred"] = (results["nm_pred"] + 1.0) / 2.0
    results["img"] = img

    saving_result(results, dst_dir)


elif args.mode == "iiw":
    assert args.iiw is not None

    input_height = 200
    input_width = 200

    (
        albedos,
        shadow,
        nm_pred,
        lighting_recon,
        shading,
        inputs,
        inputs_var,
        masks_var,
        train_flag,
    ) = irn_func(input_height, input_width)

    # load model and run session
    model_path = tf.train.get_checkpoint_state(args.model).model_checkpoint_path
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    # evaluation
    dst_dir = args.output
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir, ignore_errors=True)
    os.makedirs(dst_dir)

    iiw = args.iiw

    test_ids = np.load("utils/iiw_test_ids.npy")

    total_loss = 0
    for counter, test_id in enumerate(tqdm(test_ids)):
        img_file = str(test_id) + ".png"
        judgement_file = str(test_id) + ".json"

        img_path = os.path.join(iiw, "imgs", img_file)
        judgement_path = os.path.join(iiw, "jsons", judgement_file)

        img = io.imread(img_path)
        judgement = json.load(open(judgement_path))

        ori_img = img
        ori_height, ori_width = ori_img.shape[:2]
        img = cv2.resize(img, (input_width, input_height))
        img = np.float32(img) / 255.0
        img = img * 2.0 - 1.0
        img = img[None, :, :, :]
        mask = np.ones((1, input_height, input_width, 1), np.bool)

        [
            albedos_val,
            nm_pred_val,
            shadow_val,
            lighting_recon_val,
            shading_val,
            inputs_val,
        ] = sess.run(
            [albedos, nm_pred, shadow, lighting_recon, shading, inputs],
            feed_dict={inputs_var: img, masks_var: mask, train_flag: False},
        )

        # results folder for current scn
        result_dir = os.path.join(dst_dir, img_file[:-4])
        os.makedirs(result_dir, exist_ok=True)

        # post-process results
        results = post_process(
            albedos_val,
            shading_val,
            shadow_val,
            lighting_recon_val,
            nm_pred_val,
            ori_width,
            ori_height,
        )

        results["img"] = ori_img
        results["shading"] *= results["shadow"][..., None]
        results["nm_pred"] = (results["nm_pred"] + 1.0) / 2.0

        results["albedos"] = results["albedos"] ** (1 / 2.2)

        loss = compute_whdr(results["albedos"], judgement)
        total_loss += loss
        # print(f"{result_dir:s}\t\t{loss:f}  {total_loss:f}")

        saving_result(results, result_dir)

    print("IIW TEST WHDR %f" % (total_loss / len(test_ids)))


elif args.mode == "diode":
    assert args.diode is not None

    last_height = None
    last_width = None

    diode = args.diode
    test_root_dir = os.path.join(diode, "depth", "val")
    test_nm_root_dir = os.path.join(diode, "normal", "val")

    from glob import glob

    test_scenes_nm_dir = sorted(
        glob(os.path.join(test_nm_root_dir, "outdoor", "scene*", "scan*"))
        + glob(os.path.join(test_nm_root_dir, "indoor", "scene*", "scan*"))
    )

    test_normals_path = np.concatenate(
        [
            sorted(glob(os.path.join(t_sc_dir, "*_normal.npy")))
            for t_sc_dir in test_scenes_nm_dir
        ],
        axis=0,
    )
    test_imgs_path = np.stack(
        [
            tmp.replace("/normal/", "/depth/").replace("_normal.npy", ".png")
            for tmp in test_normals_path
        ],
        axis=0,
    )
    test_masks_path = np.stack(
        [
            tmp.replace("/normal/", "/depth/").replace("_normal.npy", "_depth_mask.npy")
            for tmp in test_normals_path
        ],
        axis=0,
    )
    test_depths_path = np.stack(
        [
            tmp.replace("/normal/", "/depth/").replace("_normal.npy", "_depth.npy")
            for tmp in test_normals_path
        ],
        axis=0,
    )

    total_angErr_list = []
    for i, (img_path, mask_path, nm_gt_path) in enumerate(
        zip(tqdm(test_imgs_path), test_masks_path, test_normals_path)
    ):

        im_id = os.path.split(img_path)[1].split(".")[0]
        cur_dir = os.path.split(img_path)[0]
        cur_dir = cur_dir.split("/val/")[1]

        # results folder
        dst_dir = args.output
        if os.path.isdir(dst_dir):
            shutil.rmtree(dst_dir, ignore_errors=True)
        os.makedirs(dst_dir)

        # load im and gts
        img = io.imread(img_path)
        ori_img = img
        img = np.float32(ori_img) / 255.0
        img = img * 2.0 - 1.0

        mask = np.load(mask_path)
        nm_gt = np.load(nm_gt_path)

        img, (ori_height, ori_width), (input_height, input_width) = rescale_img(img)
        img_mask, (_, _), (_, _) = rescale_img(mask)
        nm_gt, (_, _), (_, _) = rescale_img(nm_gt)

        img = img[None, :, :, :]
        img_mask = img_mask[None, :, :, None] != 0.0

        if input_height != last_height or input_width != last_width:
            (
                albedos,
                shadow,
                nm_pred,
                lighting_recon,
                shading,
                inputs,
                inputs_var,
                masks_var,
                train_flag,
            ) = irn_func(input_height, input_width)

            if last_height is None and last_width is None:
                model_path = tf.train.get_checkpoint_state(
                    args.model
                ).model_checkpoint_path
                sess = tf.InteractiveSession()
                saver = tf.train.Saver()
                saver.restore(sess, model_path)

            last_height = input_height
            last_width = input_width

        [
            albedos_val,
            nm_pred_val,
            shadow_val,
            lighting_recon_val,
            shading_val,
            inputs_val,
        ] = sess.run(
            [albedos, nm_pred, shadow, lighting_recon, shading, inputs],
            feed_dict={inputs_var: img, masks_var: img_mask, train_flag: False},
        )

        # results folder for current scn
        cur_dst_dir = os.path.join(dst_dir, cur_dir)
        os.makedirs(cur_dst_dir, exist_ok=True)

        # post-process results
        results = post_process(
            albedos_val,
            shading_val,
            shadow_val,
            lighting_recon_val,
            nm_pred_val,
            ori_width,
            ori_height,
            resize=False,
        )

        angErr_list = angular_error(nm_gt, results["nm_pred"])
        # print(f"{i:d}  {angErr_list.mean():f}")

        total_angErr_list.append(angErr_list)
        total_angErr_list = [np.concatenate(total_angErr_list, -1)]

        results["img"] = cv2.resize(ori_img, (input_width, input_height))
        results["nm_pred"] = (results["nm_pred"] + 1.0) / 2.0

        saving_result(results, cur_dst_dir, prefix=im_id)

    print(
        f"DIODE TEST: mean={np.mean(total_angErr_list):f}  median={np.median(total_angErr_list):.4f}"
    )
