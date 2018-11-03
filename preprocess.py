import pandas as pd
from scipy import ndimage as nd
import configure
from utils.utils import *
from utils.extract_patches import *

CFG = configure.cfg
LOG_DIR = CFG['log_file']

MIN_IMG_BOUND = -100.0  # Everything below: Water  -62   -200
MAX_IMG_BOUND = 200.0  # Everything above corresponds to bones  238    200
MIN_MSK_BOUND = 0.0  # Everything above corresponds
MAX_MSK_BOUND = 2.0  # Everything above corresponds


def set_bounds(img, MIN_BOUND, MAX_BOUND):
    image = np.clip(img, MIN_BOUND, MAX_BOUND)
    image = image.astype(np.float32)
    return image


def process_liver_tumor_nii_file(csv_path, liver_patch=False, size=224, superior=10,
                                 inferior=10):
    df = pd.read_csv(csv_path)
    patches_per_image = 500
    patch_size = 32
    imgs, masks = [], []
    for idx, item in df.iterrows():
        img = nib.load(item[0]).get_data()
        msk = nib.load(item[1]).get_data()
        img = set_bounds(img, MIN_IMG_BOUND, MAX_IMG_BOUND)
        img = normalize_with_mean(img)
        print(item)
        if liver_patch:
            minx, maxx, miny, maxy, minz, maxz = min_max_voi_with_liver(msk, superior=superior, inferior=inferior)
            img = img[minx: maxx, miny: maxy, minz: maxz]
            msk = msk[minx: maxx, miny: maxy, minz: maxz]
            scan_shape = np.array(img.shape, dtype=np.float32)
            new_shape = np.array([size, size, scan_shape[-1]], dtype=np.float32)
            resize_factor = new_shape / [scan_shape[0], scan_shape[1], scan_shape[2]]
            img = nd.interpolation.zoom(img, resize_factor, mode='nearest')
            msk = nd.interpolation.zoom(msk, resize_factor, mode='nearest')
        if not liver_patch:
            if np.unique(msk).size == 2:
                continue

        imgs.append(img)
        masks.append(msk)
        # liver_tumor_generator(patches_per_image, patch_size, img, mask, idx,
        #                       rootdir="data/patches/tumor_" + str(patch_size) + "_" + str(patches_per_image) + "/")
    # get_data_with_random_224(imgs, masks, 32)
    get_liver_tumor_data(imgs,
                         masks,
                         128,
                         128, patch_size,
                         300,
                         50,
                         fcn=True)


def process_dicom_file(csv_path):
    imgs, msks = [], []
    df = pd.read_csv(csv_path)
    for i, item in df.iterrows():
        img = read_dicom_series(item[0])
        mask = read_liver_lesion_masks(item[1])
        print('train data select dicom file:' + item[0])
        img = set_bounds(img, MIN_IMG_BOUND, MAX_IMG_BOUND)
        mask = set_bounds(mask, MIN_MSK_BOUND, MAX_MSK_BOUND)
        imgs.append(img)
        msks.append(mask)


def process_test_nii_file(csv_path):
    imgs, msks = [], []
    df = pd.read_csv(csv_path)
    for i, item in df.iterrows():
        imgnii = nib.load(item[0])
        print('train data select nii file:' + item[0])
        img = imgnii.get_data()
        img = set_bounds(img, MIN_IMG_BOUND, MAX_IMG_BOUND)
        imgs.append(img)


def resample(img, seg, scan, new_voxel_dim=[1, 1, 1]):
    # Get voxel size
    voxel_dim = np.array(scan.header.structarr["pixdim"][1:4], dtype=np.float32)
    # Resample to optimal [1,1,1] voxel size
    resize_factor = voxel_dim / new_voxel_dim
    scan_shape = np.array(scan.header.get_data_shape())
    new_scan_shape = scan_shape * resize_factor
    rounded_new_scan_shape = np.round(new_scan_shape)
    rounded_resize_factor = rounded_new_scan_shape / scan_shape  # Change resizing due to round off error
    new_voxel_dim = voxel_dim / rounded_resize_factor

    img = nd.interpolation.zoom(img, rounded_resize_factor, mode='nearest')
    seg = nd.interpolation.zoom(seg, rounded_resize_factor, mode='nearest')
    return img, seg, new_voxel_dim


def min_max_voi_with_liver(mask, superior=10, inferior=10):
    sp = mask.shape
    tp = np.transpose(np.nonzero(mask))
    minx, miny, minz = np.min(tp, axis=0)
    maxx, maxy, maxz = np.max(tp, axis=0)
    minz = 0 if minz - superior < 0 else minz - superior
    maxz = sp[-1] if maxz + inferior > sp[-1] else maxz + inferior + 1
    miny = 0 if miny - superior < 0 else miny - superior
    maxy = sp[1] if maxy + inferior > sp[1] else maxy + inferior + 1
    minx = 0 if minx - superior < 0 else minx - superior
    maxx = sp[0] if maxx + inferior > sp[0] else maxx + inferior + 1
    return minx, maxx, miny, maxy, minz, maxz


def generate_nii_from_dicom():
    # generate nii from dicom
    # import dicom2nifti
    for i in range(1, 21):
        lbl = read_liver_lesion_masks(
            'data/3Dircadb1/3Dircadb1.' + str(i) + '/MASKS_DICOM')
        imgnii = nib.load(
            'data/3Dircadb1/3Dircadb1.' + str(i) + '/3Dircadb1.' + str(
                i) + '.nii')
        img = imgnii.get_data()
        sp = img.shape

        nib.save(nib.Nifti1Image(lbl, affine=imgnii.get_affine()),
                 'data/3Dircadb1/3Dircadb1.' + str(
                     i) + '/3Dircadb_gt_1.' + str(i) + '.nii')


def preprocess_mri_nii(file_path='../datasets/brain/MICCAI_BraTS_2018_Data_Training/HGG/', ):
    image_paths = []
    patches_per_image = 500
    patch_size = 64
    image_size = (240, 240, 155)
    rootdir = "data/patches/tumor_" + str(patch_size) + "_" + str(patches_per_image) + "/"
    if not os.path.isdir(rootdir):
        os.mkdir(rootdir)
    if not os.path.isdir(rootdir + "raw"):
        os.mkdir(rootdir + "raw")
    if not os.path.isdir(rootdir + "seg"):
        os.mkdir(rootdir + "seg")

    base_locs = generate_patch_locations(patches_per_image, patch_size, image_size)
    x, y, z = perturb_patch_locations(base_locs, patch_size / 16)
    files = os.listdir(file_path)
    for name in files:
        image_paths.append(file_path + name + '/')
    file_path = file_path.replace('HGG', 'LGG')
    files = os.listdir(file_path)
    for name in files:
        image_paths.append(file_path + name + '/')
    random.shuffle(image_paths)
    vol = 0
    for item in image_paths:
        # print(i)
        print(item)
        probs = generate_patch_probs(item, (x, y, z), patch_size, image_size)
        selections = np.random.choice(range(len(probs)), size=patches_per_image, replace=False, p=probs)
        image = read_image(item)

        for num, sel in enumerate(selections):
            i, j, k = np.unravel_index(sel, (len(x), len(y), len(z)))
            patch = image[int(x[i] - patch_size / 2): int(x[i] + patch_size / 2),
                    int(y[j] - patch_size / 2): int(y[j] + patch_size / 2),
                    int(z[k] - patch_size / 2): int(z[k] + patch_size / 2), :]
            np.save(
                rootdir + 'raw/img-' + '{0:0>3}'.format(vol) + '_' + '{0:0>3}'.format(
                    num) + '.npy', patch[:, :, :, 0:4])
            np.save(
                rootdir + 'seg/msk-' + '{0:0>3}'.format(vol) + '_' + '{0:0>3}'.format(
                    num) + '.npy',
                np.expand_dims(np.where(patch[:, :, :, 4] > 0, 1, 0), -1))
        vol += 1


if __name__ == '__main__':
    root = 'Demo/idx-train-full.csv'
    process_liver_tumor_nii_file(root)
