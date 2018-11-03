import tensorflow as tf
from keras.models import model_from_json
from scipy import signal

from preprocess import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
IDEX = 0


def interpolate_pred(raws_gt, pr, kernel):
    dsfactor = [w / float(f) for w, f in zip(raws_gt.shape, pr.shape)]
    predre = nd.interpolation.zoom(255 * pr.astype('float'), zoom=dsfactor)
    jj = np.where(predre > 200)
    predre[jj] = 255
    jj = np.where(predre <= 200)
    predre[jj] = 0
    # save pred nii files in the raw shape
    predre = signal.convolve(predre, kernel, mode="same")
    jj = np.where(predre > 1)
    predre[jj] = 1
    jj = np.where(predre < 1)
    predre[jj] = 0
    return predre


def liver_boundarybox_inference(model, image_file, im_shape=(256, 256), pred_threhold=0.5):
    global IDEX
    scan_shape = np.array(image_file.shape, dtype=np.float32)
    new_shape = np.array([im_shape[0], im_shape[0], scan_shape[-1]], dtype=np.float32)
    resize_factor = new_shape / [scan_shape[0], scan_shape[1], scan_shape[2]]
    infer_image = nd.interpolation.zoom(image_file, resize_factor, mode='nearest')
    infer_image = np.expand_dims(infer_image, -1)
    infer_image = np.transpose(infer_image, (2, 0, 1, 3))
    pred = model.predict(infer_image, batch_size=20)
    pred = pred[..., 0] > pred_threhold
    pred = label_connected_component(pred)
    seg = largest_label_volume(pred)
    pred[np.where(pred != seg)] = 0
    pred[np.where(pred > 0)] = 1.
    pred = np.transpose(pred, (1, 2, 0))
    pred = nd.interpolation.zoom(pred, 1 / resize_factor, mode='nearest')
    pred[np.where(pred > 0.2)] = 1.
    minx, maxx, miny, maxy, minz, maxz = min_max_voi_with_liver(pred, superior=10, inferior=10)
    min_max_box = (int(minx), int(maxx), int(miny),
                   int(maxy), minz, maxz)
    return pred, min_max_box


def liver_voi_inference_5fold(model, boundaried_file, cube_shape=(224, 224, 32), step=(20, 20, 20),
                              pred_threhold=0.6):
    scan_shape = np.array(boundaried_file.shape, dtype=np.float32)
    new_shape = np.array([cube_shape[0], cube_shape[1], scan_shape[-1]], dtype=np.float32)
    resize_factor = new_shape / scan_shape
    boundaried_file = nd.interpolation.zoom(boundaried_file, resize_factor, mode='nearest')
    liver_pred_containers = np.zeros(boundaried_file.shape)
    sp = boundaried_file.shape
    zstep = step[2]  # 16
    deep_slices = np.arange(cube_shape[2] // 2, sp[2] - cube_shape[2] // 2 + zstep, zstep)
    if deep_slices.size == 0 or sp[-1] < cube_shape[-1]:
        raise Exception('narrow shape')
    for mi in range(len(model)):
        count_used = np.zeros(sp) + 1e-5
        liver_pred_container = np.zeros(boundaried_file.shape)
        for i in range(len(deep_slices)):
            deep = deep_slices[i]
            deep = deep if deep + cube_shape[2] // 2 <= sp[2] else -(cube_shape[2] // 2 - sp[2])
            raw_patches = boundaried_file[:, :,
                          deep - cube_shape[2] // 2:deep + cube_shape[2] // 2]
            raw_patches = raw_patches.astype(np.float32)
            X = []
            X.append(raw_patches)
            X = np.expand_dims(X, -1)
            # temp_predic = model[mi].predict(X, batch_size=1)[0, ..., 0] > pred_threhold
            temp_predic = model[mi].predict(X, batch_size=1)[0, ..., 0]
            # Major voting https://github.com/ginobilinie/infantSeg
            # currLabelMat = np.where(temp_predic == True, 1, 0)  # true, vote for 1, otherwise 0
            liver_pred_container[:, :, deep - cube_shape[2] // 2:deep + cube_shape[2] // 2] += temp_predic

            count_used[:, :, deep - cube_shape[2] // 2:deep + cube_shape[2] // 2] += 1
        # =============
        # np.save("Demo/Predictions/test/img-preds-final-3D-" + str(i) + ".npy", model.predict(X, batch_size=1))
        # np.save("Demo/Predictions/test/img-raw-3D-" + str(i) + ".npy", X)
        # =============
        # np.save("Demo/Predictions/test/img-atten4-3D-" + str(i) + ".npy",
        #         plot_layer_outputs(X, 622, model))
        # np.save("Demo/Predictions/test/img-atten3-3D-" + str(i) + ".npy",
        #         plot_layer_outputs(X, 592, model))
        # np.save("Demo/Predictions/test/img-atten2-3D-" + str(i) + ".npy",
        #         plot_layer_outputs(X, 536, model))
        # =============
        liver_pred_container = liver_pred_container / count_used
        liver_pred_containers += liver_pred_container
    # liver_pred_container = np.where(liver_pred_container > 0.5, 1, 0)
    # liver_pred_container = label_connected_component(liver_pred_container)
    # seg = largest_label_volume(liver_pred_container)
    # liver_pred_container = np.where(liver_pred_container != seg, 0, 1)
    liver_pred_container = liver_pred_containers / len(model)
    liver_pred_container = np.where(liver_pred_container > pred_threhold, 1, 0)
    # liver_pred_container = np.where(liver_pred_container > (2 / len(model)), 1, 0)

    resize_factor = scan_shape / new_shape
    liver_pred_container = nd.interpolation.zoom(liver_pred_container, resize_factor, mode='nearest')
    liver_pred_container = np.where(liver_pred_container < 0.2, 0, 1)
    # liver_pred_container = sm.binary_erosion(liver_pred_container, sm.ball(5))
    liver_pred_container = label_connected_component(liver_pred_container)
    seg = largest_label_volume(liver_pred_container)
    liver_pred_container = np.where(liver_pred_container != seg, 0, 1)
    # liver_pred_container = sm.binary_dilation(liver_pred_container, sm.ball(5))
    return liver_pred_container


def liver_tumor_inference(model, whole_img, pred_liver_tumor, cube_shape=(64, 64, 64),
                          step=(55, 55, 55), pred_threhold=0.5):
    minx, maxx, miny, maxy, minz, maxz = min_max_voi_with_liver(pred_liver_tumor, superior=10, inferior=10)
    liver_voi = whole_img[minx:maxx, miny:maxy, minz:maxz]
    pred_liver_voi = pred_liver_tumor[minx:maxx, miny:maxy, minz:maxz]
    patch_size = cube_shape
    patch_stride = 2
    tumor_pred_comtainers = np.zeros(pred_liver_voi.shape)
    locations, padding = generate_test_locations(patch_size, patch_stride, liver_voi.shape)
    pad_image = np.pad(np.expand_dims(liver_voi, -1), padding + ((0, 0),), 'constant')
    pad_result = np.zeros((pad_image.shape[:-1]), dtype=np.float32)
    pad_add = np.zeros((pad_image.shape[:-1]), dtype=np.float32)
    for mi in range(len(model)):
        for x in locations[0]:
            for y in locations[1]:
                for z in locations[2]:
                    patch = pad_image[int(x - patch_size[0] / 2): int(x + patch_size[0] / 2),
                            int(y - patch_size[1] / 2): int(y + patch_size[1] / 2),
                            int(z - patch_size[2] / 2): int(z + patch_size[2] / 2), :]

                    patch = np.expand_dims(patch, axis=0)
                    probs = model[mi].predict(patch, batch_size=1)[0, ..., 0]
                    pad_result[int(x - patch_size[0] / 2): int(x + patch_size[0] / 2),
                    int(y - patch_size[1] / 2): int(y + patch_size[1] / 2),
                    int(z - patch_size[2] / 2): int(z + patch_size[2] / 2)] += probs
                    pad_add[int(x - patch_size[0] / 2): int(x + patch_size[0] / 2),
                    int(y - patch_size[1] / 2): int(y + patch_size[1] / 2),
                    int(z - patch_size[2] / 2): int(z + patch_size[2] / 2)] += 1
        pad_result = pad_result / pad_add
        result = pad_result[padding[0][0]: padding[0][0] + liver_voi.shape[0],
                 padding[1][0]: padding[1][0] + liver_voi.shape[1],
                 padding[2][0]: padding[2][0] + liver_voi.shape[2]]
        tumor_pred_comtainers += result
    tumor_pred_container = tumor_pred_comtainers / len(model)
    tumor_pred_container = np.where(tumor_pred_container >= pred_threhold, 1, 0)
    tumor_idx = np.where(tumor_pred_container > 0)
    no_liver_idx = np.where(pred_liver_voi == 0)
    pred_liver_voi[tumor_idx] = 2
    pred_liver_voi[no_liver_idx] = 0
    pred_liver_tumor[minx:maxx, miny:maxy, minz:maxz] = pred_liver_voi
    return pred_liver_tumor, (int(minx), int(maxx), int(miny),
                              int(maxy), minz, maxz)


def infer_slice_entrance(u_model, whole_img, whole_msk, index):
    pred_img, min_max_box = liver_boundarybox_inference(u_model, whole_img)

    return pred_img, min_max_box


def infer_voi_entrance(g_model, whole_img, whole_msk, index):
    pred_img = liver_voi_inference_5fold(g_model, whole_img)
    return pred_img


def infer_tumor_entrance(c_model, whole_img, pred_img, whole_msk, index=0, pred_threhold=0.6, cube_shape=(64, 64, 64),
                         step=(55, 55, 55)):
    pred_img, min_max_box = liver_tumor_inference(c_model, whole_img, pred_img,
                                                  cube_shape=cube_shape,
                                                  step=step, pred_threhold=pred_threhold)

    return pred_img


def infer_liver(csv_path, out_path, out_csv_path):
    out_csv_path = out_csv_path + '_liver.csv'
    u_model = 'Demo/log/liver_att_resunet_2d_good_order/model_0.289/g_0.289.json'
    u_weight = 'Demo/log/liver_att_resunet_2d_good_order/model_at_epoch_00053.hdf5'
    with open(u_model, 'r') as f:
        # u_model = model_from_json(f.read())
        from build_model import get_net
        u_model = get_net((256, 256, 1), 'liver_att_resunet_2d')
        # u_model.summary()
    u_model.load_weights(u_weight)

    g_models = []

    g_model = 'Demo/log/liver_att_resunet_3d/model_0.2/g_0.2.json'
    g_weight = 'Demo/log/liver_att_resunet_3d/model_at_2018-10-27_01:07:28_epoch_00049_f1.hdf5'
    with open(g_model, 'r') as f:
        g_model = model_from_json(f.read())
    g_model.load_weights(g_weight)
    g_models.append(g_model)

    g_model = 'Demo/log/liver_att_resunet_3d/model_0.2/g_0.2.json'
    g_weight = 'Demo/log/liver_att_resunet_3d/model_at_2018-10-29_20:45:12_epoch_00049_f2.hdf5'
    with open(g_model, 'r') as f:
        g_model = model_from_json(f.read())
    g_model.load_weights(g_weight)
    g_models.append(g_model)

    g_model = 'Demo/log/liver_att_resunet_3d/model_0.2/g_0.2.json'
    g_weight = 'Demo/log/liver_att_resunet_3d/model_at_2018-10-30_08:43:05_epoch_00047_f3.hdf5'
    with open(g_model, 'r') as f:
        g_model = model_from_json(f.read())
    g_model.load_weights(g_weight)
    g_models.append(g_model)

    g_model = 'Demo/log/liver_att_resunet_3d/model_0.2/g_0.2.json'
    g_weight = 'Demo/log/liver_att_resunet_3d/model_at_2018-11-02_08:41:20_epoch_00049_f4.hdf5'
    with open(g_model, 'r') as f:
        g_model = model_from_json(f.read())
    g_model.load_weights(g_weight)
    g_models.append(g_model)

    g_model = 'Demo/log/liver_att_resunet_3d/model_0.2/g_0.2.json'
    g_weight = 'Demo/log/liver_att_resunet_3d/model_at_2018-11-02_00:47:25_epoch_00036.hdf5'
    with open(g_model, 'r') as f:
        g_model = model_from_json(f.read())
    g_model.load_weights(g_weight)
    g_models.append(g_model)

    df = pd.read_csv(csv_path)
    cvscores = []
    for i, item in df.iterrows():
        print('load raw select file:' + item[0])
        # train
        imgnii = nib.load(item[0])
        whole_img = imgnii.get_data()
        whole_img = set_bounds(whole_img, MIN_IMG_BOUND, MAX_IMG_BOUND)
        whole_img = normalize_with_mean(whole_img)
        try:
            whole_msk = nib.load(item[1]).get_data()
        except Exception as err:
            print(err)
            whole_msk = None
        pred_img, min_max_box = infer_slice_entrance(u_model, whole_img, whole_msk, i)
        print(min_max_box)
        # min_max_box1 = min_max_voi_with_liver(whole_msk)
        # print(min_max_box1)
        # print(np.array(min_max_box) - np.array(min_max_box1))
        liver_voi = whole_img[min_max_box[0]: min_max_box[1], min_max_box[2]: min_max_box[3], min_max_box[4]:
                                                                                              min_max_box[5]]
        if whole_msk is None:
            msk_voi = None
        else:
            msk_voi = whole_msk[min_max_box[0]: min_max_box[1], min_max_box[2]: min_max_box[3], min_max_box[4]:
                                                                                                min_max_box[5]]

        try:
            pred_img = infer_voi_entrance(g_models, liver_voi, msk_voi, i)
        except Exception as err:
            print(err)
            continue

        pred_container = np.zeros(whole_img.shape)
        pred_container[min_max_box[0]: min_max_box[1], min_max_box[2]: min_max_box[3],
        min_max_box[4]:min_max_box[5]] = pred_img

        # pred_container = pred_img
        if whole_msk is not None:
            msk_img_path = item[1]
            msknii = nib.load(msk_img_path)
            # print(np.unique(pred_container))
            liver_scores = scorer(pred_container >= 1, msknii.get_data() >= 1,
                                  msknii.header.get_zooms()[:3])
            print("Liver dice", liver_scores['dice'])
            cvscores.append(liver_scores['dice'])

            outstr = str(i) + ','
            for l in [liver_scores]:
                for k, v in l.items():
                    outstr += str(v) + ','
                outstr += '\n'

            if not os.path.isfile(out_csv_path):
                headerstr = 'Volume,'
                for k, v in liver_scores.items():
                    headerstr += 'Liver_' + k + ','
                headerstr += '\n'
                outstr = headerstr + outstr
            f = open(out_csv_path, 'a+')
            f.write(outstr)
            f.close()
        nib.save(nib.Nifti1Image(pred_container.astype('float'), affine=imgnii.get_affine()),
                 out_path + 'test-segmentation-' + str(i) + '.nii.gz')
    print("%.4f%%" % (np.mean(cvscores)))


def infer_tumor(csv_path, out_path, out_csv_path):
    out_csv_path = out_csv_path + '_tumor.csv'
    c_models = []

    c_model = 'Demo/log/liver_tumor_res_att_unet_3d/model_0.2/g_0.2.json'
    c_weight = 'Demo/log/liver_tumor_res_att_unet_3d/model_at_2018-10-21_01:06:21_epoch_00083.hdf5'
    with open(c_model, 'r') as f:
        c_model = model_from_json(f.read())
    c_model.load_weights(c_weight)
    c_models.append(c_model)

    c_model = 'Demo/log/liver_tumor_res_att_unet_3d/model_0.2/g_0.2.json'
    c_weight = 'Demo/log/liver_tumor_res_att_unet_3d/model_at_2018-10-21_09:28:37_epoch_00041.hdf5'
    with open(c_model, 'r') as f:
        c_model = model_from_json(f.read())
    c_model.load_weights(c_weight)
    c_models.append(c_model)

    c_model = 'Demo/log/liver_tumor_res_att_unet_3d/model_0.2/g_0.2.json'
    c_weight = 'Demo/log/liver_tumor_res_att_unet_3d/model_at_2018-10-22_11:50:41_epoch_00046.hdf5'
    with open(c_model, 'r') as f:
        c_model = model_from_json(f.read())
    c_model.load_weights(c_weight)
    c_models.append(c_model)

    c_model = 'Demo/log/liver_tumor_res_att_unet_3d/model_0.2/g_0.2.json'
    c_weight = 'Demo/log/liver_tumor_res_att_unet_3d/model_at_2018-07-06_13:24:50_epoch_00099.hdf5'
    with open(c_model, 'r') as f:
        c_model = model_from_json(f.read())
    c_model.load_weights(c_weight)
    c_models.append(c_model)

    c_model = 'Demo/log/liver_tumor_res_att_unet_3d/model_0.2/g_0.2.json'
    c_weight = 'Demo/log/liver_tumor_res_att_unet_3d/model_at_2018-10-23_07:56:48_epoch_00047.hdf5'
    with open(c_model, 'r') as f:
        c_model = model_from_json(f.read())
    c_model.load_weights(c_weight)
    c_models.append(c_model)

    out_path = out_path + 'tumor/'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    df = pd.read_csv(csv_path)
    for i, item in df.iterrows():
        print('load raw select file:' + item[0])
        print('load pred liver container file:' + item[2])
        # train
        # whole_img = np.load(item[0])
        whole_img = nib.load(item[0]).get_data()
        whole_img = set_bounds(whole_img, MIN_IMG_BOUND, MAX_IMG_BOUND)
        whole_img = normalize_with_mean(whole_img)

        try:
            whole_msk = nib.load(item[1]).get_data()
        except Exception as err:
            print(err)
            whole_msk = None
        pred_container_nii = nib.load(item[2])
        pred_container = pred_container_nii.get_data()
        # std, mean = cal_mean(whole_img)
        # whole_img = normalize(whole_img, mean, std)
        pred_container = infer_tumor_entrance(c_models, whole_img, pred_container, whole_msk, i, pred_threhold=0.5,
                                              cube_shape=(64, 64, 64),
                                              step=(55, 55, 55))
        if whole_msk is not None:
            tumor_scores = scorer(pred_container > 1, whole_msk > 1,
                                  pred_container_nii.header.get_zooms()[:3])
            print("Tumor dice", tumor_scores['dice'])
            if tumor_scores['dice'] < 0.02:
                continue
            outstr = str(i) + ','
            for l in [tumor_scores]:
                for k, v in l.items():
                    outstr += str(v) + ','
                outstr += '\n'

            if not os.path.isfile(out_csv_path):
                headerstr = 'Volume,'
                for k, v in tumor_scores.items():
                    headerstr += 'Tumor_' + k + ','
                headerstr += '\n'
                outstr = headerstr + outstr
            f = open(out_csv_path, 'a+')
            f.write(outstr)
        nib.save(nib.Nifti1Image(pred_container.astype('float'), affine=pred_container_nii.get_affine()),
                 out_path + 'test-segmentation-' + str(i) + '.nii.gz')


def infer_Brain_Tumor(out_path, rootdir="/root/share/datasets/brain/Brats17ValidationData/"):
    c_models = []

    c_model = 'Demo/log/brain_tumor_res_atten_unet_3d/model_0.2/g_0.2.json'
    c_weight = 'Demo/log/brain_tumor_res_atten_unet_3d/model_at_2018-10-10_05:47:51_epoch_00049.hdf5'
    with open(c_model, 'r') as f:
        c_model = model_from_json(f.read())
    c_model.load_weights(c_weight)
    c_models.append(c_model)

    if "17" in rootdir:
        out_path = out_path + 'tumor/2017/'
    elif "18" in rootdir:
        out_path = out_path + 'tumor/2018/'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    files = os.listdir(rootdir)
    patch_size = (64, 64, 64)
    patch_stride = 4
    for name in files:
        path = rootdir + name + '/'
        print('load raw select file:' + name)
        image = read_image(path, is_training=False)
        locations, padding = generate_test_locations(patch_size, patch_stride, image.shape[:-1])
        pad_image = np.pad(image, padding + ((0, 0),), 'constant')
        pad_result = np.zeros((pad_image.shape[:-1]), dtype=np.float32)
        pad_add = np.zeros((pad_image.shape[:-1]), dtype=np.float32)
        for x in locations[0]:
            for y in locations[1]:
                for z in locations[2]:
                    patch = pad_image[int(x - patch_size[0] / 2): int(x + patch_size[0] / 2),
                            int(y - patch_size[1] / 2): int(y + patch_size[1] / 2),
                            int(z - patch_size[2] / 2): int(z + patch_size[2] / 2), :]

                    patch = np.expand_dims(patch, axis=0)

                    probs = c_models[0].predict(patch, batch_size=1)[0, ..., 0]

                    pad_result[int(x - patch_size[0] / 2): int(x + patch_size[0] / 2),
                    int(y - patch_size[1] / 2): int(y + patch_size[1] / 2),
                    int(z - patch_size[2] / 2): int(z + patch_size[2] / 2)] += probs
                    pad_add[int(x - patch_size[0] / 2): int(x + patch_size[0] / 2),
                    int(y - patch_size[1] / 2): int(y + patch_size[1] / 2),
                    int(z - patch_size[2] / 2): int(z + patch_size[2] / 2)] += 1
        pad_result = pad_result / pad_add
        result = pad_result[padding[0][0]: padding[0][0] + image.shape[0],
                 padding[1][0]: padding[1][0] + image.shape[1],
                 padding[2][0]: padding[2][0] + image.shape[2]]
        pred = np.where(result > 0.5, 1, 0)
        # pred = clean_contour(result, pred)
        # print(path)
        # pred_container = np.where(whole_img > 0, 1, 0)
        nib.save(nib.Nifti1Image(pred.astype('float'), np.eye(4)),
                 out_path + name + '.nii.gz')


def infer_Brain_TCIA_Tumor(out_path, rootdir="../datasets/brain/MICCAI_BraTS_2018_Data_Training/HGG/"):
    c_models = []
    from keras.models import Model
    c_model = 'Demo/log/brain_tumor_res_atten_unet_3d/model_0.2/g_0.2.json'
    c_weight = 'Demo/log/brain_tumor_res_atten_unet_3d/model_at_2018-10-11_13:58:37_epoch_00010.hdf5'
    with open(c_model, 'r') as f:
        c_model = model_from_json(f.read())
    c_model.load_weights(c_weight)
    c_models.append(c_model)

    out_path = out_path + 'TCIA/'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    image_paths = []
    files = os.listdir(rootdir)
    for name in files:
        image_paths.append(rootdir + name + '/')
    file_path = rootdir.replace('HGG', 'LGG')
    files = os.listdir(file_path)
    for name in files:
        image_paths.append(file_path + name + '/')
    files = image_paths
    cube_shape = (64, 64, 64)
    step = (60, 60, 60)
    files = sorted(files)
    intermediate_layer_model = Model(c_models[0].input,
                                     c_models[0].layers[639].output)
    for ff in range(len(files)):
        name = files[ff]
        if "TCIA" not in name:
            continue
        print(ff)
        print('load raw select file:' + name)
        image = read_image(name, is_training=True)
        fileName = name[name.rfind('/Brats') + 1:len(name) - 1]
        print('load raw select file:' + fileName)
        mask = image[..., 4]
        image = image[..., 0:4]
        minx, maxx, miny, maxy, minz, maxz = min_max_voi_with_liver(mask, superior=20, inferior=20)
        mask = mask[minx:maxx, miny:maxy, minz:maxz]
        image = image[minx:maxx, miny:maxy, minz:maxz, :]
        imagec = np.zeros((image.shape[0], image.shape[1], image.shape[2], 5))
        imagec[..., 0:4] = image
        imagec[..., 4] = mask
        image = imagec
        sp = image.shape
        xstep = step[0]  # 16
        ystep = step[1]  # 16
        zstep = step[2]  # 16
        width_slices = np.arange(cube_shape[0] // 2, sp[0] - cube_shape[0] // 2 + xstep, xstep)
        height_slices = np.arange(cube_shape[1] // 2, sp[1] - cube_shape[1] // 2 + ystep, ystep)
        deep_slices = np.arange(cube_shape[2] // 2, sp[2] - cube_shape[2] // 2 + zstep, zstep)
        print(len(width_slices) * len(deep_slices) * len(height_slices))
        for i in range(len(deep_slices)):
            deep = deep_slices[i]
            deep = deep if deep + cube_shape[2] // 2 <= sp[2] else -(cube_shape[2] // 2 - sp[2])
            for j in range(len(height_slices)):
                height = height_slices[j]
                height = height if height + cube_shape[1] // 2 <= sp[1] else -(cube_shape[1] // 2 - sp[1])
                for k in range(len(width_slices)):
                    width = width_slices[k]
                    width = width if width + cube_shape[0] // 2 <= sp[0] else -(cube_shape[0] // 2 - sp[0])
                    patch = image[width - cube_shape[0] // 2:width + cube_shape[0] // 2,
                            height - cube_shape[1] // 2:height + cube_shape[1] // 2,
                            deep - cube_shape[2] // 2:deep + cube_shape[2] // 2]
                    msk = patch[..., 4]
                    patch = patch[..., 0:4]
                    patch = np.expand_dims(patch, axis=0)
                    # 635 concate   636 conv 32number  639 activation later
                    intermediate_output = intermediate_layer_model.predict(patch, batch_size=1)[0, ...]
                    # out_list = get_i_layer_outputs(patch, c_models[0], 639)
                    features = np.concatenate([intermediate_output, np.expand_dims(msk, -1)], axis=-1)
                    np.save(out_path + fileName + str(i) + '_' + str(j) + '_' + str(k) + ".npy", features)


if __name__ == '__main__':

    data_dir = 'lits_test'
    if data_dir == 'lits_validation':
        csv_path = 'Demo/idx-test-lits-npy.csv'
        out_path = 'Demo/Predictions/Liver/LiTS_validation/'
        out_csv_path = 'Demo/Predictions/LiTS_validation_results'
    elif data_dir == 'lits_test':
        csv_path = 'Demo/idx-test-full.csv'
        out_path = 'Demo/Predictions/Liver/LiTS_test/'
        out_csv_path = 'Demo/Predictions/LiTS_test_results'
    elif data_dir == '3DIRCADb':
        csv_path = 'Demo/idx-3Dircadb1-full.csv'
        out_path = 'Demo/Predictions/Liver/3DIRCADb/'
        out_csv_path = 'Demo/Predictions/3DIRCADb_results'
    elif data_dir == 'Brain':
        out_path = 'Demo/Predictions/Brain/'
    # infer_liver(csv_path, out_path, out_csv_path)
    if data_dir == 'Brain':
        infer_Brain_Tumor(out_path)
        # infer_Brain_TCIA_Tumor(out_path)
    else:
        infer_tumor(csv_path, out_path, out_csv_path)
