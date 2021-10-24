import pickle as pk
import os
import numpy as np
import tensorflow as tf
import skimage.transform as imgTform
import glob
from scipy import io


def bigTime_dataPipeline(num_subbatch_input, dir):
    img_batch = pk.load(open(os.path.join(dir + "BigTime_v1", "img_batch.p"), "rb"))

    scene = "0161"
    img_batch = [sorted(sc_batch) for sc_batch in img_batch if scene in sc_batch[0][:4]]
    img_batch = np.asarray(img_batch[0])
    img_batch = [np.delete(img_batch, [10, 20, 30, 40, 50])]

    sc_len = np.asarray([len(sc_l) for sc_l in img_batch], dtype=np.int32)
    size_dim2 = sc_len.max()

    bt_imgs_path = []
    bt_masks_path = []
    for sc_list, sc_size in zip(img_batch, sc_len):
        sc_imgs_path = []
        sc_masks_path = []
        for img_string in sc_list:
            tmp = img_string.split(os.path.sep)

            img_path = os.path.join(dir + "BigTime_v1", tmp[0], "data", tmp[1])

            img_name = os.path.splitext(tmp[1])[0]
            mask_path = os.path.join(
                dir + "BigTime_v1", tmp[0], "data", img_name + "_mask.png"
            )

            sc_imgs_path.append(img_path)
            sc_masks_path.append(mask_path)

        sc_imgs_path += ["" for i in range(size_dim2 - sc_size)]
        sc_masks_path += ["" for i in range(size_dim2 - sc_size)]
        bt_imgs_path.append(sc_imgs_path)
        bt_masks_path.append(sc_masks_path)

    bt_imgs_path = np.asarray(bt_imgs_path)
    bt_masks_path = np.asarray(bt_masks_path)

    train_data = bt_construct_inputPipeline(
        bt_imgs_path,
        bt_masks_path,
        sc_len,
        batch_size=num_subbatch_input,
        flag_shuffle=True,
    )

    # define re-initialisable iterator
    iterator = tf.data.Iterator.from_structure(
        train_data.output_types, train_data.output_shapes
    )
    next_element = iterator.get_next()

    # define initialisation for each iterator
    trainData_init_op = iterator.make_initializer(train_data)

    return next_element, trainData_init_op, len(bt_imgs_path)


def megaDepth_dataPipeline(num_subbatch_input, dir, training_mode, num_test_sc):
    # locate all scenes
    data_scenes1 = np.array(
        sorted(glob.glob(os.path.join(dir + "new_outdoorMega_items", "*")))
    )
    data_scenes2 = np.array(
        sorted(glob.glob(os.path.join(dir + "new_indoorMega_items", "*")))
    )
    data_scenes3 = np.array(
        sorted(glob.glob(os.path.join(dir + "new_LSMega_items", "*")))
    )

    # scan scenes
    # sort scenes by number of training images in each
    scenes_size1 = np.array([len(os.listdir(i)) for i in data_scenes1])
    scenes_size2 = np.array([len(os.listdir(i)) for i in data_scenes2])
    scenes_size3 = np.array([len(os.listdir(i)) for i in data_scenes3])
    scenes_sorted1 = np.argsort(scenes_size1)
    scenes_sorted2 = np.argsort(scenes_size2)
    scenes_sorted3 = np.argsort(scenes_size3)

    train_scenes = data_scenes1[scenes_sorted1[num_test_sc:]]
    test_scenes = data_scenes1[scenes_sorted1[:num_test_sc]]

    cProj_HiRes_scenes = np.array(
        [
            os.path.join(dir + "HiRes_cProj_imgs", sc.split("/")[-1])
            for sc in train_scenes
        ]
    )
    cProj_HiRes_test_scenes = np.array(
        [
            os.path.join(dir + "HiRes_cProj_imgs", sc.split("/")[-1])
            for sc in test_scenes
        ]
    )

    # load data from each scene
    # locate each data minibatch in each sorted sc
    train_scenes_items = [
        sorted(glob.glob(os.path.join(sc, "*.pk"))) for sc in train_scenes
    ]
    train_scenes_items = np.concatenate(train_scenes_items, axis=0)
    test_scenes_items = [
        sorted(glob.glob(os.path.join(sc, "*.pk"))) for sc in test_scenes
    ]
    test_scenes_items = np.concatenate(test_scenes_items, axis=0)

    HiRes_cProj_items = [
        sorted(glob.glob(os.path.join(sc, "*.pk"))) for sc in cProj_HiRes_scenes
    ]
    HiRes_cProj_items = np.concatenate(HiRes_cProj_items, axis=0)
    HiRes_cProj_test_items = [
        sorted(glob.glob(os.path.join(sc, "*.pk"))) for sc in cProj_HiRes_test_scenes
    ]
    HiRes_cProj_test_items = np.concatenate(HiRes_cProj_test_items, axis=0)

    # split data into train and test
    # separate out some data from training scenes as testing data
    train_items = train_scenes_items
    test_items = test_scenes_items

    ### contruct training data pipeline
    # remove residual data over number of data in one epoch
    res_train_items = len(train_items) - (len(train_items) % num_subbatch_input)
    train_items = train_items[:res_train_items]
    HiRes_cProj_items = HiRes_cProj_items[:res_train_items]
    train_data = md_construct_inputPipeline(
        train_items, HiRes_cProj_items, flag_shuffle=True, batch_size=num_subbatch_input
    )

    res_test_items = len(test_items) - (len(test_items) % num_subbatch_input)
    test_items = test_items[:res_test_items]
    HiRes_cProj_test_items = HiRes_cProj_test_items[:res_test_items]
    test_data = md_construct_inputPipeline(
        test_items,
        HiRes_cProj_test_items,
        flag_shuffle=False,
        batch_size=num_subbatch_input,
    )

    # define re-initialisable iterator
    iterator = tf.data.Iterator.from_structure(
        train_data.output_types, train_data.output_shapes
    )
    next_element = iterator.get_next()

    # define initialisation for each iterator
    trainData_init_op = iterator.make_initializer(train_data)
    testData_init_op = iterator.make_initializer(test_data)

    return (
        next_element,
        trainData_init_op,
        testData_init_op,
        len(train_items),
        len(test_items),
    )


def nyu_dataPipeline(num_subbatch_input, dir):
    nm_gts_path = np.array(
        glob.glob(os.path.join(dir, "normals_gt", "new_normals", "*"))
    )
    nm_gts_path.sort()
    masks_path = np.array(glob.glob(os.path.join(dir, "normals_gt", "masks", "*")))
    masks_path.sort()
    splits_path = os.path.join(dir, "splits.mat")
    imgs_path = os.path.join(dir, "NYU_imgs.mat")
    train_split = io.loadmat(splits_path)["trainNdxs"]
    train_split -= 1
    train_split = train_split.squeeze()
    test_split = io.loadmat(splits_path)["testNdxs"]
    test_split -= 1
    test_split = test_split.squeeze()
    train_split = np.squeeze(train_split)
    imgs = io.loadmat(imgs_path)["imgs"]
    imgs = imgs.transpose(-1, 0, 1, 2)

    train_nm_gts_path = nm_gts_path[train_split]
    train_masks_path = masks_path[train_split]
    train_imgs = imgs[train_split]

    train_data = nyu_construct_inputPipeline(
        train_imgs,
        train_nm_gts_path,
        train_masks_path,
        batch_size=num_subbatch_input,
        flag_shuffle=True,
    )

    # define re-initialisable iterator
    iterator = tf.data.Iterator.from_structure(
        train_data.output_types, train_data.output_shapes
    )
    next_element = iterator.get_next()

    # define initialisation for each iterator
    trainData_init_op = iterator.make_initializer(train_data)

    return next_element, trainData_init_op


def _read_pk_function(filename):
    with open(filename, "rb") as f:
        batch_data = pk.load(f)
    input = np.float32(batch_data["input"])
    dm = batch_data["dm"]
    nm = np.float32(batch_data["nm"])
    cam = np.float32(batch_data["cam"])
    scaleX = batch_data["scaleX"]
    scaleY = batch_data["scaleY"]
    mask = np.float32(batch_data["mask"])

    return input, dm, nm, cam, scaleX, scaleY, mask


def _read_pk_function_cProj(filename):
    with open(filename, "rb") as f:
        batch_data = pk.load(f)
    input = np.float32(batch_data["reproj_im1"])
    mask = np.float32(batch_data["reproj_mask"])

    return input, mask


def md_read_func(filename, cProj_filename):

    input, dm, nm, cam, scaleX, scaleY, mask = tf.py_func(
        _read_pk_function,
        [filename],
        [
            tf.float32,
            tf.float32,
            tf.float32,
            tf.float32,
            tf.float32,
            tf.float32,
            tf.float32,
        ],
    )
    reproj_inputs, reproj_mask = tf.py_func(
        _read_pk_function_cProj, [cProj_filename], [tf.float32, tf.float32]
    )

    input = tf.data.Dataset.from_tensor_slices(input[None])
    dm = tf.data.Dataset.from_tensor_slices(dm[None])
    nm = tf.data.Dataset.from_tensor_slices(nm[None])
    cam = tf.data.Dataset.from_tensor_slices(cam[None])
    scaleX = tf.data.Dataset.from_tensor_slices(scaleX[None])
    scaleY = tf.data.Dataset.from_tensor_slices(scaleY[None])
    mask = tf.data.Dataset.from_tensor_slices(mask[None])
    reproj_inputs = tf.data.Dataset.from_tensor_slices(reproj_inputs[None])
    reproj_mask = tf.data.Dataset.from_tensor_slices(reproj_mask[None])

    return tf.data.Dataset.zip(
        (input, dm, nm, cam, scaleX, scaleY, mask, reproj_inputs, reproj_mask)
    )


def md_preprocess_func(
    input, dm, nm, cam, scaleX, scaleY, mask, reproj_inputs, reproj_mask
):

    input = input / 255
    input = input * 2 - 1

    nm = nm / 127

    reproj_inputs = reproj_inputs / 255

    reproj_inputs = reproj_inputs * 2 - 1

    return input, dm, nm, cam, scaleX, scaleY, mask, reproj_inputs, reproj_mask


def bt_preprocess_func(bt_imgs, bt_masks):
    ori_bt_imgs = tf.identity(bt_imgs)
    ori_bt_masks = tf.identity(bt_masks)

    input_height = 200
    input_width = 200

    ori_height = tf.shape(bt_imgs)[1]
    ori_width = tf.shape(bt_imgs)[2]
    ratio = tf.to_float(ori_width) / tf.to_float(ori_height)

    bt_imgs = tf.image.resize_nearest_neighbor(bt_imgs, (input_height, input_width))
    bt_masks = tf.image.resize_nearest_neighbor(bt_masks, (input_height, input_width))

    bt_imgs = tf.to_float(bt_imgs) / 255.0
    bt_masks = tf.to_float(tf.not_equal(bt_masks, 0))

    return bt_imgs, bt_masks


def bt_read_func(bt_imgs_path, bt_masks_path, sc_len):
    batch_size = tf.constant(5)
    res_len = batch_size - sc_len
    res_idx = tf.range(sc_len, batch_size)

    sfl_idx = tf.random_shuffle(tf.range(sc_len))

    sc_idx = sfl_idx[:batch_size]
    bt_imgs_path = tf.gather(bt_imgs_path, sc_idx)
    bt_masks_path = tf.gather(bt_masks_path, sc_idx)

    i_ = tf.constant(0)
    num_loops = tf.shape(bt_imgs_path)[0]
    im_output = tf.TensorArray(dtype=tf.uint8, size=num_loops)
    mask_output = tf.TensorArray(dtype=tf.uint8, size=num_loops)

    def condition(i_, im_output, mask_output):
        return tf.less(i_, num_loops)

    def body(i_, im_output, mask_output):
        bt_img = tf.read_file(bt_imgs_path[i_])
        bt_img = tf.image.decode_image(bt_img)

        bt_mask = tf.read_file(bt_masks_path[i_])
        bt_mask = tf.image.decode_image(bt_mask)

        im_output = im_output.write(i_, bt_img)
        mask_output = mask_output.write(i_, bt_mask)
        i_ += 1

        return i_, im_output, mask_output

    _, im_output, mask_output = tf.while_loop(
        condition, body, loop_vars=[i_, im_output, mask_output]
    )

    bt_imgs = im_output.stack()[tf.newaxis]
    bt_masks = mask_output.stack()[tf.newaxis]

    return tf.data.Dataset.from_tensor_slices((bt_imgs, bt_masks))


def nyu_read_func(img, nm_gt_path, mask_path):

    nm_gt = tf.image.decode_image(tf.read_file(nm_gt_path), channels=3)
    mask = tf.image.decode_image(tf.read_file(mask_path))

    return tf.data.Dataset.from_tensor_slices(
        (img[tf.newaxis, :, :, :], tf.expand_dims(nm_gt, axis=0), mask[tf.newaxis])
    )


def nyu_preprocess_func(img, nm_gt, mask):

    # masking
    bdL = tf.reduce_min(tf.where(tf.not_equal(mask, 0))[:, 0])
    bdR = tf.reduce_max(tf.where(tf.not_equal(mask, 0))[:, 0])
    bdT = tf.reduce_min(tf.where(tf.not_equal(mask, 0))[:, 1])
    bdB = tf.reduce_max(tf.where(tf.not_equal(mask, 0))[:, 1])

    img = img[bdT:bdB, bdL:bdR]
    nm_gt = nm_gt[bdT:bdB, bdL:bdR]

    img = tf.to_float(img) / 255.0
    nm_gt = tf.to_float(nm_gt) / 127.0 - 1.0

    nm_gt = tf.stack([nm_gt[:, :, 2], -nm_gt[:, :, 1], -nm_gt[:, :, 0]], axis=-1)

    img = img[tf.newaxis]
    nm_gt = nm_gt[tf.newaxis]

    input_height = 200
    input_width = 200

    ori_height = tf.shape(img)[1]
    ori_width = tf.shape(img)[2]
    ratio = tf.to_float(ori_width) / tf.to_float(ori_height)

    rand_pos = tf.cond(
        ratio > 1.0,
        lambda: f1(ori_height, ori_width),
        lambda: f2(ori_height, ori_width),
    )

    rand_flip = tf.random_uniform((), 0, 1, dtype=tf.float32)
    rand_angle = tf.random_uniform((), -1, 1, dtype=tf.float32) * (5.0 / 180.0) * np.pi

    img = img[:, rand_pos[0] : rand_pos[1], rand_pos[2] : rand_pos[3], :]
    nm_gt = nm_gt[:, rand_pos[0] : rand_pos[1], rand_pos[2] : rand_pos[3], :]

    img = tf.where(rand_flip > 0.5, img[:, :, ::-1], img)
    nm_gt = tf.where(
        rand_flip > 0.5,
        nm_gt[:, :, ::-1] * tf.constant([[[[-1, 1, 1]]]], dtype=tf.float32),
        nm_gt,
    )

    img = tf.image.resize_nearest_neighbor(img, (input_height, input_width))
    nm_gt = tf.image.resize_nearest_neighbor(nm_gt, (input_height, input_width))

    img = tf.contrib.image.rotate(img, rand_angle)
    nm_gt = tf.contrib.image.rotate(nm_gt, rand_angle)

    sinR = tf.sin(rand_angle)
    cosR = tf.cos(rand_angle)
    R = tf.stack(
        [
            tf.stack([cosR, sinR, 0.0], axis=-1),
            tf.stack([-sinR, cosR, 0.0], axis=-1),
            tf.constant([0, 0, 1], dtype=tf.float32),
        ],
        axis=0,
    )
    nm_gt = tf.reshape(tf.matmul(tf.reshape(nm_gt, (-1, 3)), R), (1, 200, 200, 3))

    return tf.squeeze(img), tf.squeeze(nm_gt)


def bt_construct_inputPipeline(
    bt_imgs_path, bt_masks_path, sc_len, batch_size, flag_shuffle=True
):
    data = tf.data.Dataset.from_tensor_slices((bt_imgs_path, bt_masks_path, sc_len))
    if flag_shuffle:
        data = data.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=100000))
    else:
        data = data.repeat()

    data = data.apply(
        tf.contrib.data.parallel_interleave(
            bt_read_func, cycle_length=batch_size, block_length=1, sloppy=False
        )
    )

    data = data.map(bt_preprocess_func, num_parallel_calls=8)
    data = data.batch(batch_size).prefetch(4)

    return data


def nyu_construct_inputPipeline(
    nyu_imgs, nyu_nm_gts_path, nyu_masks_path, batch_size, flag_shuffle=True
):
    imgs_data = tf.data.Dataset.from_tensor_slices(nyu_imgs)
    nm_gts_data = tf.data.Dataset.from_tensor_slices(nyu_nm_gts_path)
    masks_data = tf.data.Dataset.from_tensor_slices(nyu_masks_path)

    data = tf.data.Dataset.zip((imgs_data, nm_gts_data, masks_data))

    if flag_shuffle:
        data = data.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1000))
    else:
        data = data.repeat()
    data = data.apply(
        tf.contrib.data.parallel_interleave(
            nyu_read_func, cycle_length=batch_size, block_length=1, sloppy=False
        )
    )

    data = data.map(nyu_preprocess_func, num_parallel_calls=8)
    data = data.batch(batch_size).prefetch(4)

    return data


def md_construct_inputPipeline(items, cProj_items, batch_size, flag_shuffle=True):
    data = tf.data.Dataset.from_tensor_slices((items, cProj_items))
    if flag_shuffle:
        data = data.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=100000))
    else:
        data = data.repeat()
    data = data.apply(
        tf.contrib.data.parallel_interleave(
            md_read_func, cycle_length=batch_size, block_length=1, sloppy=False
        )
    )
    data = data.map(md_preprocess_func, num_parallel_calls=8)
    data = data.batch(batch_size).prefetch(4)

    return data


def f1(ori_h, ori_w):
    h_upMost = 25
    w_leftMost = ori_w - ori_h + h_upMost

    random_start_y = tf.random_uniform((), 0, h_upMost, dtype=tf.int32)
    random_start_x = tf.random_uniform((), 0, w_leftMost, dtype=tf.int32)
    random_pos = [
        random_start_y,
        random_start_y + ori_h - h_upMost,
        random_start_x,
        random_start_x + ori_w - w_leftMost,
    ]

    return random_pos


def f2(ori_h, ori_w):
    w_leftMost = 25
    h_upMost = ori_h - ori_w + w_leftMost

    random_start_x = tf.random_uniform((), 0, w_leftMost, dtype=tf.int32)
    random_start_y = tf.random_uniform((), 0, h_upMost, dtype=tf.int32)
    random_pos = [
        random_start_y,
        random_start_y + ori_h - h_upMost,
        random_start_x,
        random_start_x + ori_w - w_leftMost,
    ]

    return random_pos
