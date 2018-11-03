import numpy as np
from keras import backend as K
import os
import nibabel as nib
import sys
import scipy.misc
from keras.preprocessing.image import Iterator
from scipy.ndimage import rotate
from medpy import metric
from surface import Surface
import random
from glob import glob
from skimage import measure

IMG_DTYPE = np.float
SEG_DTYPE = np.uint8

import dicom
import natsort
import re


# ====================================================
# ======================volume preprocessing method===
# ====================================================
def to_scale(img, slice_shape, shape=None):
    if shape is None:
        shape = slice_shape

    height, width = shape
    if img.dtype == SEG_DTYPE:
        return scipy.misc.imresize(img, (height, width), interp="nearest").astype(SEG_DTYPE)
    elif img.dtype == IMG_DTYPE:
        factor = 256.0 / np.max(img)
        return (scipy.misc.imresize(img, (height, width), interp="nearest") / factor).astype(IMG_DTYPE)
    else:
        raise TypeError(
            'Error. To scale the image array, its type must be np.uint8 or np.float64. (' + str(img.dtype) + ')')


def norm_hounsfield_dyn(arr, c_min=0.1, c_max=0.3):
    """ Converts from hounsfield units to float64 image with range 0.0 to 1.0 """
    # calc min and max
    min, max = np.amin(arr), np.amax(arr)
    if min <= 0:
        arr = np.clip(arr, min * c_min, max * c_max)
        # right shift to zero
        arr = np.abs(min * c_min) + arr
    else:
        arr = np.clip(arr, min, max * c_max)
        # left shift to zero
        arr = arr - min
    # normalization
    norm_fac = np.amax(arr)
    if norm_fac != 0:
        norm = np.divide(
            np.multiply(arr, 255),
            np.amax(arr))
    else:  # don't divide through 0
        norm = np.multiply(arr, 255)

    norm = np.clip(np.multiply(norm, 0.00390625), 0, 1)
    return norm


def histeq_processor(img):
    """Histogram equalization"""
    nbr_bins = 256
    # get image histogram
    imhist, bins = np.histogram(img.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize
    # use linear interpolation of cdf to find new pixel values
    original_shape = img.shape
    img = np.interp(img.flatten(), bins[:-1], cdf)
    img = img / 256.0
    return img.reshape(original_shape)


def read_dicom_series(directory, filepattern="image_*"):
    """ Reads a DICOM Series files in the given directory.
    Only filesnames matching filepattern will be considered"""

    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise ValueError("Given directory does not exist or is a file : " + str(directory))
    print('\tRead Dicom', directory)
    lstFilesDCM = natsort.natsorted(glob(os.path.join(directory, filepattern)))
    print('\tLength dicom series', len(lstFilesDCM))
    # Get ref file
    RefDs = dicom.read_file(lstFilesDCM[0])
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = dicom.read_file(filenameDCM)
        # store the raw image data
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

    return ArrayDicom


def rotate(matrix):
    """
    :type matrix: List[List[int]]
    :rtype: void Do not return anything, modify matrix in-place instead.
    """
    matrix[:] = map(list, zip(*matrix[::-1]))


def read_liver_lesion_masks(masks_dirname):
    """Since 3DIRCAD provides an individual mask for each tissue type (in DICOM series format),
    we merge multiple tissue types into one Tumor mask, and merge this mask with the liver mask

    Args:
        masks_dirname : MASKS_DICOM directory containing multiple DICOM series directories,
                        one for each labelled mask
    Returns:
        numpy array with 0's for background pixels, 1's for liver pixels and 2's for tumor pixels
    """
    tumor_volume = None
    liver_volume = None

    # For each relevant organ in the current volume
    for organ in os.listdir(masks_dirname):
        organ_path = os.path.join(masks_dirname, organ)
        if not os.path.isdir(organ_path):
            continue

        organ = organ.lower()

        if organ.startswith("livertumor") or re.match("liver.yst.*", organ) or organ.startswith(
                "stone") or organ.startswith("metastasecto"):
            print('Organ', masks_dirname, organ)
            current_tumor = read_dicom_series(organ_path)
            current_tumor = np.clip(current_tumor, 0, 1)
            # Merge different tumor masks into a single mask volume
            tumor_volume = current_tumor if tumor_volume is None else np.logical_or(tumor_volume, current_tumor)
        elif organ == 'liver':
            print('Organ', masks_dirname, organ)
            liver_volume = read_dicom_series(organ_path)
            liver_volume = np.clip(liver_volume, 0, 1)

    # Merge liver and tumor into 1 volume with background=0, liver=1, tumor=2
    label_volume = np.zeros(liver_volume.shape)
    label_volume[liver_volume == 1] = 1
    label_volume[tumor_volume == 1] = 2
    label_final = np.zeros(label_volume.shape)
    for j in range(label_volume.shape[-1]):
        im = label_volume[:, :, j]
        rotate(im)
        label_final[:, :, j] = im
    return label_final


# Get majority label in image
def largest_label_volume(img, bg=0):
    vals, counts = np.unique(img, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def label_connected_component(pred):
    seg = measure.label(pred, neighbors=8, background=0)
    return seg


def scorer(pred, label, vxlspacing):
    volscores = {}

    volscores['dice'] = metric.dc(pred, label)
    volscores['jaccard'] = metric.binary.jc(pred, label)
    volscores['voe'] = 1. - volscores['jaccard']
    volscores['rvd'] = metric.ravd(label, pred)

    if np.count_nonzero(pred) == 0 or np.count_nonzero(label) == 0:
        volscores['assd'] = 0
        volscores['msd'] = 0
    else:
        evalsurf = Surface(pred, label, physical_voxel_spacing=vxlspacing, mask_offset=[0., 0., 0.],
                           reference_offset=[0., 0., 0.])
        volscores['assd'] = evalsurf.get_average_symmetric_surface_distance()

        volscores['msd'] = metric.hd(label, pred, voxelspacing=vxlspacing)

    return volscores


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def normalize_with_mean(img):
    imgs_normalized = (img - np.mean(img)) / np.std(img)
    # imgs_normalized =img
    imgs_normalized = (
            (imgs_normalized - np.min(imgs_normalized)) / (np.max(imgs_normalized) - np.min(imgs_normalized)))
    return imgs_normalized


def define_log(basic_path, exp):
    LOG_FILE = basic_path + '/' + exp
    if not os.path.exists(LOG_FILE):
        print("DIRECTORY Created")
        os.makedirs(LOG_FILE)
    f = open(LOG_FILE + '/' + exp + '_sys_out.log', 'a')
    sys.stdout = Tee(sys.stdout, f)

    # handler = logging.handlers.RotatingFileHandler(LOG_FILE + '/' + exp + '.log', maxBytes=50 * 1024 * 1024,
    #                                                backupCount=5)
    # fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(message)s'
    # formatter = logging.Formatter(fmt)
    # handler.setFormatter(formatter)
    # logger = logging.getLogger(LOG_FILE)
    # logger.addHandler(handler)
    # logger.setLevel(logging.DEBUG)
    # return logger


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        pass


def get_layer_outputs(test_image, model):
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    comp_graph = [K.function([model.input] + [K.learning_phase()], [output]) for output in
                  outputs]  # evaluation functions

    # Testing
    comp_graph = comp_graph[1:len(comp_graph)]
    layer_outputs_list = [op([test_image, 1.]) for op in comp_graph]
    layer_outputs = []

    for layer_output in layer_outputs_list:
        # print(layer_output[0][0].shape, end='\n-------------------\n')
        layer_outputs.append(layer_output[0][0])

    return layer_outputs


def get_i_layer_outputs(test_image, model, i):
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    comp_graph = [K.function([model.input] + [K.learning_phase()], [output]) for output in
                  outputs]  # evaluation functions
    # Testing
    comp_graph = comp_graph[1:len(comp_graph)]
    layer_output = comp_graph[i - 1]([test_image, 1.])[0][0]
    return layer_output


def plot_layer_outputs(test_image, layer_number, model, c=3):
    layer_outputs = get_layer_outputs(test_image, model)

    x_max = layer_outputs[layer_number].shape[0]
    y_max = layer_outputs[layer_number].shape[1]
    n = layer_outputs[layer_number].shape[2]

    L = []
    for i in range(n):
        L.append(np.zeros((x_max, y_max)))

    for i in range(n):
        for x in range(x_max):
            for y in range(y_max):
                L[i][x][y] = layer_outputs[layer_number][x][y][i]

    # for img in L:
    #     plt.figure()
    #     plt.imshow(img, interpolation='nearest')
    return L


class TrainBatchFetcher(Iterator):
    """
    fetch batch of original images and liver images
    """

    def __init__(self, data_dir, valid_fraction, valid_from=0):
        # self.train_imgs = train_imgs
        # self.train_livers = train_livers
        # self.n_train_imgs = self.train_imgs.shape[0]
        images = data_dir + '/raw/img-*.npy'
        labels = data_dir + '/seg/msk-*.npy'
        image_paths = []
        for name in sorted(glob(images), reverse=True):
            image_paths.append(name)
        label_paths = {
            os.path.basename(path).replace('msk-', 'img-'): path
            for path in sorted(glob(labels), reverse=True)}
        self.label_paths = label_paths

        num_images = len(image_paths)
        # random.shuffle(image_paths)
        if num_images == 0:
            raise RuntimeError('No data files found in ' + data_dir)

        self.valid_images = image_paths[int(valid_from * num_images):int(valid_fraction * num_images)]
        # print(self.valid_images[0])
        # print(self.valid_images[len(self.valid_images)-1])
        # self.train_images = image_paths[int(valid_fraction * num_images):]
        self.train_images = list(set(image_paths).difference(set(self.valid_images)))
        random.shuffle(self.valid_images)
        random.shuffle(self.train_images)
        self.num_training = len(self.train_images)
        self.num_validation = len(self.valid_images)
        print('number of training :' + str(self.num_training))
        print('number of validation :' + str(self.num_validation))

    def next(self, batch_size, type='train'):
        # indices = list(np.random.choice(self.n_train_imgs, batch_size))
        # return self.train_imgs[indices, :, :, :], self.train_livers[indices, :, :, :]
        image_paths = self.train_images
        num_training = self.num_training
        if type != 'train':
            image_paths = self.valid_images
            num_training = self.num_validation
            print('vali training' + str(num_training))

        while (1):
            random.shuffle(image_paths)
            indices = list(np.random.choice(num_training, batch_size))
            images = []
            labels = []
            for index in indices:
                # label_file = np.load(self.label_paths[os.path.basename(self.train_images[index])])
                # image_file = np.load(self.train_images[index])
                label_file = np.load(self.label_paths[os.path.basename(image_paths[index])])
                image_file = np.load(image_paths[index])
                images.append(image_file)
                labels.append(label_file)
                # im = array_to_img(image_file)
                # im.save('Demo/c_slices_prediction/raw_slice_vol_img/vol_img_' + str(index) + '.jpg', 'jpeg')
                # im = array_to_img(label_file)
                # im.save('Demo/c_slices_prediction/msk_slice_vol_img/vol_img_' + str(index) + '.jpg',
                #         'jpeg')
            images = np.array(images)
            labels = np.array(labels)
            # labels = np.concatenate((1 - labels, labels), -1)
            yield images, labels

    def nextVali(self, batch_size):
        image_paths = self.valid_images
        while (1):
            random.shuffle(image_paths)
            indices = list(np.random.choice(self.num_validation, batch_size))
            images = []
            labels = []
            for index in indices:
                label_file = np.load(self.label_paths[os.path.basename(self.valid_images[index])])
                image_file = np.load(self.valid_images[index])
                images.append(image_file)
                labels.append(label_file)
            images = np.array(images)
            labels = np.array(labels)
            yield images, labels

    def vali_data(self):
        image_paths = self.valid_images
        random.shuffle(image_paths)
        images = []
        labels = []
        for image_file in image_paths:
            label_file = np.load(self.label_paths[os.path.basename(image_file)])
            image_file = np.load(image_file)  # image_file
            images.append(image_file)
            labels.append(label_file)
        images = np.array(images)
        labels = np.array(labels)
        # labels = np.concatenate((1 - labels, labels), -1)
        return images, labels

    def get_one(self):
        image_paths = self.valid_images
        random.shuffle(image_paths)
        images = []
        labels = []
        label_file = np.load(self.label_paths[os.path.basename(image_paths[0])])
        image_file = np.load(image_paths[0])  # image_file
        images.append(image_file)
        labels.append(label_file)
        images = np.array(images)
        labels = np.array(labels)
        return images, labels


# brain utils
def generate_patch_locations(patches, patch_size, im_size):
    nx = round((patches * 8 * im_size[0] * im_size[0] / im_size[1] / im_size[2]) ** (1.0 / 3))
    ny = round(nx * im_size[1] / im_size[0])
    nz = round(nx * im_size[2] / im_size[0])
    x = np.rint(np.linspace(patch_size, im_size[0] - patch_size, num=nx))
    y = np.rint(np.linspace(patch_size, im_size[1] - patch_size, num=ny))
    z = np.rint(np.linspace(patch_size, im_size[2] - patch_size, num=nz))
    return x, y, z


def perturb_patch_locations(patch_locations, radius):
    x, y, z = patch_locations
    x = np.rint(x + np.random.uniform(-radius, radius, len(x)))
    y = np.rint(y + np.random.uniform(-radius, radius, len(y)))
    z = np.rint(z + np.random.uniform(-radius, radius, len(z)))
    return x, y, z


def generate_patch_probs(path, patch_locations, patch_size, im_size, tagTumor=0):
    x, y, z = patch_locations
    seg = nib.load(glob(os.path.join(path, '*_seg.nii.gz'))[0]).get_data().astype(np.float32)
    p = []
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                patch = seg[int(x[i] - patch_size / 2): int(x[i] + patch_size / 2),
                        int(y[j] - patch_size / 2): int(y[j] + patch_size / 2),
                        int(z[k] - patch_size / 2): int(z[k] + patch_size / 2)]
                patch = (patch > tagTumor).astype(np.float32)
                percent = np.sum(patch) / (patch_size * patch_size * patch_size)
                p.append((1 - np.abs(percent - 0.5)) * percent)
    p = np.asarray(p, dtype=np.float32)
    p[p == 0] = np.amin(p[np.nonzero(p)])
    p = p / np.sum(p)
    return p


def normalize(im_input):
    superior = 10
    inferior = 10
    sp = im_input.shape
    tp = np.transpose(np.nonzero(im_input))
    minx, miny, minz = np.min(tp, axis=0)
    maxx, maxy, maxz = np.max(tp, axis=0)
    minz = 0 if minz - superior < 0 else minz - superior
    maxz = sp[-1] if maxz + inferior > sp[-1] else maxz + inferior + 1
    miny = 0 if miny - superior < 0 else miny - superior
    maxy = sp[1] if maxy + inferior > sp[1] else maxy + inferior + 1
    minx = 0 if minx - superior < 0 else minx - superior
    maxx = sp[0] if maxx + inferior > sp[0] else maxx + inferior + 1

    roi = im_input[minx: maxx, miny: maxy, minz: maxz]
    # im_output = (im_input - np.mean(roi)) / np.std(roi)
    im_output = (im_input - np.min(roi)) / (np.max(roi) - np.min(roi))
    return im_output


def read_image(path, is_training=True):
    t1 = nib.load(glob(os.path.join(path, '*_t1.nii.gz'))[0]).get_data().astype(np.float32)
    t1ce = nib.load(glob(os.path.join(path, '*_t1ce.nii.gz'))[0]).get_data().astype(np.float32)
    t2 = nib.load(glob(os.path.join(path, '*_t2.nii.gz'))[0]).get_data().astype(np.float32)
    flair = nib.load(glob(os.path.join(path, '*_flair.nii.gz'))[0]).get_data().astype(np.float32)
    assert t1.shape == t1ce.shape == t2.shape == flair.shape
    if is_training:
        seg = nib.load(glob(os.path.join(path, '*_seg.nii.gz'))[0]).get_data().astype(np.float32)
        assert t1.shape == seg.shape
        nchannel = 5
    else:
        nchannel = 4

    image = np.empty((t1.shape[0], t1.shape[1], t1.shape[2], nchannel), dtype=np.float32)

    # image[..., 0] = remove_low_high(t1)
    # image[..., 1] = remove_low_high(t1ce)
    # image[..., 2] = remove_low_high(t2)
    # image[..., 3] = remove_low_high(flair)
    image[..., 0] = normalize(t1)
    image[..., 1] = normalize(t1ce)
    image[..., 2] = normalize(t2)
    image[..., 3] = normalize(flair)

    if is_training:
        image[..., 4] = seg

    return image


def generate_test_locations(patch_size, stride, im_size):
    stride_size_x = patch_size[0] / stride
    stride_size_y = patch_size[1] / stride
    stride_size_z = patch_size[2] / stride
    pad_x = (
        int(patch_size[0] / 2),
        int(np.ceil(im_size[0] / stride_size_x) * stride_size_x - im_size[0] + patch_size[0] / 2))
    pad_y = (
        int(patch_size[1] / 2),
        int(np.ceil(im_size[1] / stride_size_y) * stride_size_y - im_size[1] + patch_size[1] / 2))
    pad_z = (
        int(patch_size[2] / 2),
        int(np.ceil(im_size[2] / stride_size_z) * stride_size_z - im_size[2] + patch_size[2] / 2))
    x = np.arange(patch_size[0] / 2, im_size[0] + pad_x[0] + pad_x[1] - patch_size[0] / 2 + 1, stride_size_x)
    y = np.arange(patch_size[1] / 2, im_size[1] + pad_y[0] + pad_y[1] - patch_size[1] / 2 + 1, stride_size_y)
    z = np.arange(patch_size[2] / 2, im_size[2] + pad_z[0] + pad_z[1] - patch_size[2] / 2 + 1, stride_size_z)
    return (x, y, z), (pad_x, pad_y, pad_z)
