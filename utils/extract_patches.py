import numpy as np
import random
import os
from utils.utils import generate_patch_locations, perturb_patch_locations
from keras.utils import np_utils

np.random.seed(1337)


def liver_tumor_generator(patches_per_image, patch_size, img, mask, idx, rootdir="data/patches/tumor_32_400/"):
    if not os.path.isdir(rootdir):
        os.mkdir(rootdir)
    if not os.path.isdir(rootdir + "raw"):
        os.mkdir(rootdir + "raw")
    if not os.path.isdir(rootdir + "seg"):
        os.mkdir(rootdir + "seg")
    base_locs = generate_patch_locations(patches_per_image, patch_size, mask.shape)
    x, y, z = perturb_patch_locations(base_locs, patch_size / 16)
    # generate_patch_probs
    p = []
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                patch = mask[int(x[i] - patch_size / 2): int(x[i] + patch_size / 2),
                        int(y[j] - patch_size / 2): int(y[j] + patch_size / 2),
                        int(z[k] - patch_size / 2): int(z[k] + patch_size / 2)]
                patch = (patch > 1).astype(np.float32)
                percent = np.sum(patch) / (patch_size * patch_size * patch_size)
                p.append((1 - np.abs(percent - 0.5)) * percent)
    p = np.asarray(p, dtype=np.float32)
    p[p == 0] = np.amin(p[np.nonzero(p)])
    probs = p / np.sum(p)
    # generate_patch_probs
    selections = np.random.choice(range(len(probs)), size=patches_per_image, replace=False, p=probs)
    imgs_normalized = (img - np.mean(img)) / np.std(img)
    img_save = (imgs_normalized - np.min(imgs_normalized)) / (np.max(imgs_normalized) - np.min(imgs_normalized))
    for num, sel in enumerate(selections):
        i, j, k = np.unravel_index(sel, (len(x), len(y), len(z)))
        patch = img_save[int(x[i] - patch_size / 2): int(x[i] + patch_size / 2),
                int(y[j] - patch_size / 2): int(y[j] + patch_size / 2),
                int(z[k] - patch_size / 2): int(z[k] + patch_size / 2)]
        patch_msk = mask[int(x[i] - patch_size / 2): int(x[i] + patch_size / 2),
                    int(y[j] - patch_size / 2): int(y[j] + patch_size / 2),
                    int(z[k] - patch_size / 2): int(z[k] + patch_size / 2)]
        assert patch.shape == (patch_size, patch_size, patch_size)
        assert patch_msk.shape == (patch_size, patch_size, patch_size)
        np.save(rootdir + 'raw/img-' + '{0:0>3}'.format(idx) + '_' + '{0:0>3}'.format(num) + '.npy',
                np.expand_dims(patch, -1))
        np.save(rootdir + 'seg/msk-' + '{0:0>3}'.format(idx) + '_' + '{0:0>3}'.format(num) + '.npy',
                np.expand_dims(np.where(patch_msk > 1, 1, 0), -1))


def get_liver_tumor_data(train_imgs_original,
                          train_groudTruth,
                          patch_height,
                          patch_width, patch_depth,
                          N_subimgs_positive,
                          N_subimgs_negative,
                          fcn=False):
    train_masks = train_groudTruth
    train_imgs = train_imgs_original

    # extract the TRAINING patches from the full images
    patches_imgs_train, patches_masks_train = extract_random_with_balance(train_imgs, train_masks, patch_height,
                                                                          patch_width, patch_depth,
                                                                          N_subimgs_positive, N_subimgs_negative,
                                                                          fcn=fcn)
    data_consistency_check_patches(patches_imgs_train, patches_masks_train)
    ##Fourier transform of patches
    if not fcn:
        patches_masks_train = np_utils.to_categorical(patches_masks_train, 2)
    else:
        patches_masks_train = np.concatenate((1 - patches_masks_train, patches_masks_train), -1)
    # already -= in the save npy phase
    # patches_imgs_train -= np.mean(patches_imgs_train)
    # patches_imgs_train /= np.max(patches_imgs_train)
    return patches_imgs_train, patches_masks_train  # , patches_imgs_test, patches_masks_test


# extract patches randomly in the full training images
#  -- Inside OR in full image
def extract_random_with_balance(full_imgs, full_masks, patch_x, patch_y, patch_z, N_patches_positive=200,
                                N_patches_negative=100, N_patches_border=0, fcn=False, liver=1.):
    patch_per_img_positive = int(N_patches_positive)
    patch_per_img_negative = int(N_patches_negative)
    patch_per_img_border = int(N_patches_border)
    patches = []
    patches_masks = []
    rootdir = "data/patches/liver_tumor_" + str(patch_x) + "_" + str(N_patches_positive + N_patches_negative) + "/"
    if not os.path.isdir(rootdir):
        os.mkdir(rootdir)
    if not os.path.isdir(rootdir + "raw"):
        os.mkdir(rootdir + "raw")
    if not os.path.isdir(rootdir + "seg"):
        os.mkdir(rootdir + "seg")
    print(len(full_imgs))
    if not fcn:
        # Extract patch of 29*29*29 for liver tumor
        for i in range(len(full_imgs)):  # loop over the full images
            data_consistency_check(full_imgs[i], full_masks[i])
            k = 0
            idx_p = np.where(full_masks[i] > 1)
            if idx_p[0].size < 10:
                continue
            idx_n = np.where(full_masks[i] == 1.)
            # scales = math.ceil(idx_p[0].size / (patch_x * patch_y * patch_z))
            # print(scales)
            # patch_per_img_positive=scales*150
            # patch_per_img_negative=patch_per_img_positive
            print("positive patches per full image: " + str(patch_per_img_positive))
            print("negative patches per full image: " + str(patch_per_img_negative))
            while k < (patch_per_img_positive + patch_per_img_negative):
                if k < patch_per_img_positive:
                    # positive patch
                    index = np.random.randint(len(idx_p[0]), size=1)[0]
                    x_center, y_center, z_center = idx_p[0][index], idx_p[1][index], idx_p[2][index]
                else:
                    # patch_per_img_negative patch
                    index = np.random.randint(len(idx_n[0]), size=1)[0]
                    x_center, y_center, z_center = idx_n[0][index], idx_n[1][index], idx_n[2][index]
                patch = full_imgs[i][x_center - int(patch_x / 2):x_center + int(patch_x / 2) + 1,
                        y_center - int(patch_y / 2):y_center + int(patch_y / 2) + 1,
                        z_center - int(patch_z / 2):z_center + int(patch_z / 2) + 1]
                # print(patch.shape)
                if patch.shape != (patch_x, patch_y, patch_z):
                    continue
                patch_mask = full_masks[i][x_center, y_center, z_center] - 1
                patches.append(patch)
                patches_masks.append(patch_mask)
                k += 1  # per full_img
        patches = np.expand_dims(patches, -1)
        patches_masks = np.clip(np.array(patches_masks), 0, 1)
    else:
        # Extract patch of patch_x*patch_y*patch_z for liver tumor with class balance
        for i in range(len(full_imgs)):  # loop over the full images
            data_consistency_check(full_imgs[i], full_masks[i])
            idx_p = np.where(full_masks[i] > liver)
            if idx_p[0].size < 10:
                continue
            if full_imgs[i].shape[-1] < patch_z:
                continue
            k = 0
            print(i)
            # if scales < 0.5:
            #     patch_per_img_positive = 400
            # elif 0.5 < scales < 1:
            #     patch_per_img_positive = 300
            # else:
            #     patch_per_img_positive = int(N_patches_positive)
            print('=========')
            idx_n = np.where(full_masks[i] == liver)
            idx_b = np.where(full_masks[i] == 0.)
            while k < (patch_per_img_positive + patch_per_img_negative + patch_per_img_border):
                if k < patch_per_img_positive:
                    # positive patch
                    index = np.random.randint(len(idx_p[0]), size=1)[0]
                    x_center, y_center, z_center = idx_p[0][index], idx_p[1][index], idx_p[2][index]
                    # from -patch_x to patch_x contain tumor patch
                    x_center += random.randint(-patch_x / 2, patch_x / 2)
                    y_center += random.randint(-patch_y / 2, patch_y / 2)
                    z_center += random.randint(-patch_z / 2, patch_z / 2)
                elif patch_per_img_positive <= k <= patch_per_img_positive + patch_per_img_negative:
                    # patch_per_img_negative patch
                    index = np.random.randint(len(idx_n[0]), size=1)[0]
                    x_center, y_center, z_center = idx_n[0][index], idx_n[1][index], idx_n[2][index]
                else:
                    while (True):
                        index = np.random.randint(len(idx_b[0]), size=1)[0]
                        x_center, y_center, z_center = idx_b[0][index], idx_b[1][index], idx_b[2][index]
                        msk = full_masks[i][x_center - int(patch_x / 2):x_center + int(patch_x / 2),
                              y_center - int(patch_y / 2):y_center + int(patch_y / 2),
                              z_center - int(patch_z / 2):z_center + int(patch_z / 2)]
                        sizeofmsk = np.where(msk > 0)
                        if len(sizeofmsk[0]) / patch_y / patch_x / patch_z > 0.3:
                            break

                patch = full_imgs[i][x_center - int(patch_x / 2):x_center + int(patch_x / 2),
                        y_center - int(patch_y / 2):y_center + int(patch_y / 2),
                        z_center - int(patch_z / 2):z_center + int(patch_z / 2)]
                msk = full_masks[i][x_center - int(patch_x / 2):x_center + int(patch_x / 2),
                      y_center - int(patch_y / 2):y_center + int(patch_y / 2),
                      z_center - int(patch_z / 2):z_center + int(patch_z / 2)]
                # print(patch.shape)
                if patch.shape != (patch_x, patch_y, patch_z):
                    continue
                np.save(rootdir + 'raw/img-' + '{0:0>3}'.format(i) + '_' + '{0:0>3}'.format(k) + '.npy',
                        np.expand_dims(patch, -1))
                np.save(rootdir + 'seg/msk-' + '{0:0>3}'.format(i) + '_' + '{0:0>3}'.format(k) + '.npy',
                        np.expand_dims(np.where(msk > 1, 1, 0), -1))
                k += 1  # per full_img
    return patches, patches_masks


def get_data_with_random_224(train_imgs_original,
                             train_groudTruth, patch_z,
                             zstep=25):
    rootdir = "data/patches/liver_" + str(224) + "/"
    if not os.path.isdir(rootdir):
        os.mkdir(rootdir)
    if not os.path.isdir(rootdir + "raw"):
        os.mkdir(rootdir + "raw")
    if not os.path.isdir(rootdir + "seg"):
        os.mkdir(rootdir + "seg")

    full_masks = train_groudTruth
    full_imgs = train_imgs_original
    pid = 0
    # Extract patch of 64*64*64 for liver(inside of voi)
    for i in range(len(full_imgs)):  # loop over the full images
        data_consistency_check(full_imgs[i], full_masks[i])
        if full_imgs[i].shape[-1] < patch_z:
            continue
        k = 0
        patch_per_img = int(full_imgs[i].shape[-1] / 20 * 2) * 4
        while k < patch_per_img:
            z_center = np.random.randint(int(patch_z / 2), full_imgs[i].shape[-1] - int(patch_z / 2))
            patch = full_imgs[i][:,
                    :,
                    z_center - int(patch_z / 2):z_center + int(patch_z / 2)]
            msk = full_masks[i][:,
                  :,
                  z_center - int(patch_z / 2):z_center + int(patch_z / 2)]
            # print(patch.shape)
            assert (patch.shape == (224, 224, patch_z))
            # patches.append(patch)
            # patches_masks.append(np.clip(msk, 0., 1.))
            k += 1  # per full_img
            np.save(rootdir + 'raw/img-%03d' % i + '_' + str(pid) + '.npy', np.expand_dims(patch, -1))
            msk = np.where(msk < 0.5, 0, 1)
            np.save(rootdir + 'seg/msk-%03d' % i + '_' + str(pid) + '.npy', np.expand_dims(msk, -1))
            pid += 1
        deep_slices = np.arange(patch_z // 2, full_imgs[i].shape[-1] - patch_z // 2 + zstep, zstep)
        if deep_slices.size == 0 or full_imgs[i].shape[-1] < patch_z:
            continue
        for j in range(len(deep_slices)):
            deep = deep_slices[j]
            deep = deep if deep + patch_z // 2 <= full_imgs[i].shape[-1] else -(patch_z // 2 - full_imgs[i].shape[-1])
            patch = full_imgs[i][:, :,
                    deep - patch_z // 2:deep + patch_z // 2]
            msk = full_masks[i][:, :,
                  deep - patch_z // 2:deep + patch_z // 2]
            k += 1
            np.save(rootdir + 'raw/img-%03d' % i + '_' + str(pid) + '.npy', np.expand_dims(patch, -1))
            msk = np.where(msk < 0.5, 0, 1)
            np.save(rootdir + 'seg/msk-%03d' % i + '_' + str(pid) + '.npy', np.expand_dims(msk, -1))
            pid += 1


# extract patches randomly in the full validation images
#  -- Inside OR in full image
def extract_random_patches(full_imgs, full_masks, patch_x, patch_y, patch_z, N_patches, patch_voi=False):
    patch_per_img = int(N_patches)
    print("patches per full image: " + str(patch_per_img))
    patches = []
    patches_masks = []
    if not patch_voi:
        # Extract patch of 29*29*29 for liver tumor
        for i in range(len(full_imgs)):  # loop over the full images
            data_consistency_check(full_imgs[i], full_masks[i])
            k = 0
            idx = np.where(full_masks[i] >= 1.)
            while k < patch_per_img:
                index = np.random.randint(len(idx[0]), size=1)[0]
                x_center, y_center, z_center = idx[0][index], idx[1][index], idx[2][index]
                # x_center = np.random.randint(low = 0+int(patch_w/2),high = img_w-int(patch_w/2))
                # # print "x_center " +str(x_center)
                # y_center = np.random.randint(low=0 + int(patch_h / 2), high=img_h - int(patch_h / 2))
                # # print "y_center " +str(y_center)
                # z_center = np.random.randint(low=0 + int(patch_z / 2), high=img_z - int(patch_z / 2))
                patch = full_imgs[i][x_center - int(patch_x / 2):x_center + int(patch_x / 2) + 1,
                        y_center - int(patch_y / 2):y_center + int(patch_y / 2) + 1,
                        z_center - int(patch_z / 2):z_center + int(patch_z / 2) + 1]
                # print(patch.shape)
                if patch.shape != (patch_x, patch_y, patch_z):
                    continue
                assert (patch.shape == (patch_x, patch_y, patch_z))
                patch_mask = full_masks[i][x_center, y_center, z_center]
                patches.append(patch)
                patches_masks.append(patch_mask - 1)
                k += 1  # per full_img
        patches = np.expand_dims(patches, -1)
        patches_masks = np.array(patches_masks)
    else:
        # Extract patch of 64*64*64 for liver(inside of voi)
        for i in range(len(full_imgs)):  # loop over the full images
            data_consistency_check(full_imgs[i], full_masks[i])
            if full_imgs[i].shape[-1] < patch_z:
                continue
            k = 0
            idx = np.where(full_masks[i] >= 1.)
            while k < patch_per_img:
                index = np.random.randint(len(idx[0]), size=1)[0]
                x_center, y_center, z_center = idx[0][index], idx[1][index], idx[2][index]
                patch = full_imgs[i][x_center - int(patch_x / 2):x_center + int(patch_x / 2),
                        y_center - int(patch_y / 2):y_center + int(patch_y / 2),
                        z_center - int(patch_z / 2):z_center + int(patch_z / 2)]
                msk = full_masks[i][x_center - int(patch_x / 2):x_center + int(patch_x / 2),
                      y_center - int(patch_y / 2):y_center + int(patch_y / 2),
                      z_center - int(patch_z / 2):z_center + int(patch_z / 2)]
                # print(patch.shape)
                if patch.shape != (patch_x, patch_y, patch_z):
                    print(patch.shape)
                    continue
                np.save('data/patches/tumor_32_much_more/raw/img-' + '{0:0>3}'.format(i) + '_' + str(k) + '.npy',
                        np.expand_dims(patch, -1))
                np.save('data/patches/tumor_32_much_more/seg/msk-' + '{0:0>3}'.format(i) + '_' + str(k) + '.npy',
                        np.expand_dims(np.where(msk > 1, 1, 0), -1))
                print('data/patches/tumor_32_much_more/raw/img-' + '{0:0>3}'.format(i) + '_' + str(k))
                print('data/patches/tumor_32_much_more/seg/msk-' + '{0:0>3}'.format(i) + '_' + str(k))

                k += 1  # per full_img
        # patches = np.expand_dims(patches, -1)
        # patches_masks = np.expand_dims(patches_masks, -1)
    return patches, patches_masks


def data_consistency_check(imgs, masks):
    assert (len(imgs.shape) == len(masks.shape))
    assert (imgs.shape[0] == masks.shape[0])
    assert (imgs.shape[1] == masks.shape[1])
    assert (imgs.shape[2] == masks.shape[2])


def data_consistency_check_patches(imgs, masks):
    assert (len(imgs.shape) == 5)
    assert (imgs.shape[0] == masks.shape[0])
    # image.shape[1] >1 in using gabor wavelet,  so cannot have fixed number of channels
    # assert(imgs.shape[1]==1 or imgs.shape[1]==3)
