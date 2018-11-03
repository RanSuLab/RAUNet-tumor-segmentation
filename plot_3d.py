import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import nibabel as nib
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import gc
import matplotlib.cm as cm
import matplotlib
from sklearn.metrics import auc
from PIL import Image
import cv2

""""""""""""""""""""""""""""""""""""""""""""""""
""" 3D Reconstructing the picture """
""""""""""""""""""""""""""""""""""""""""""""""""


def miccaiimshow(img, seg, preds, fname, titles=None, plot_separate_img=True):
    """Takes raw image img, seg in range 0-2, list of predictions in range 0-2"""
    plt.figure(figsize=(25, 25))
    ALPHA = 1

    if len(preds.shape) == 3:
        n_plots = len(preds)
    else:
        n_plots = 1
    subplot_offset = 0

    plt.set_cmap('gray')

    if plot_separate_img:
        n_plots += 1
        subplot_offset = 1
        plt.subplot(1, n_plots, 1)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.title("Image")
        plt.axis('off')
        plt.imshow(img, cmap="gray")
    if type(preds) != list:
        preds = [preds]
    for i, pred in enumerate(preds):
        # Order of overaly
        ########## OLD
        # lesion= pred==2
        # difflesion = set_minus(seg==2,lesion)
        # liver = set_minus(pred==1, [lesion, difflesion])
        # diffliver = set_minus(seg==1, [liver,lesion,difflesion])
        ##########

        lesion = pred == 2
        difflesion = np.logical_xor(seg == 2, lesion)
        liver = pred == 1
        diffliver = np.logical_xor(seg == 1, liver)

        plt.subplot(1, n_plots, i + 1 + subplot_offset)
        title = titles[i] if titles is not None and i < len(titles) else ""
        plt.title(title)
        plt.axis('off')
        plt.imshow(img);
        plt.hold(True)
        # Liver prediction
        plt.imshow(np.ma.masked_where(liver == 0, liver), cmap="Greens", vmin=0.1, vmax=1.2, alpha=ALPHA)
        plt.hold(True)
        # Liver : Pixels in ground truth, not in prediction
        plt.imshow(np.ma.masked_where(diffliver == 0, diffliver), cmap="Spectral", vmin=0.1, vmax=2.2, alpha=ALPHA)
        plt.hold(True)

        # Lesion prediction
        plt.imshow(np.ma.masked_where(lesion == 0, lesion), cmap="Blues", vmin=0.1, vmax=1.2, alpha=ALPHA)
        plt.hold(True)
        # Lesion : Pixels in ground truth, not in prediction
        plt.imshow(np.ma.masked_where(difflesion == 0, difflesion), cmap="Reds", vmin=0.1, vmax=1.5, alpha=ALPHA)

    plt.savefig(fname, transparent=True)
    plt.close()


def plot_AUC_ROC(fprs, tprs, method_names, fig_dir, op_pts):
    # set font style
    font = {'family': 'serif'}
    matplotlib.rc('font', **font)

    # sort the order of plots manually for eye-pleasing plots
    colors = ['r', 'b', 'y', 'g', '#7e7e7e', 'm', 'c', 'k', '#cd919e'] if len(fprs) == 9 else ['r', 'y', 'm', 'g', 'k']
    indices = [7, 2, 5, 3, 4, 6, 1, 8, 0] if len(fprs) == 9 else [4, 1, 2, 3, 0]

    # print auc
    print("****** ROC AUC ******")
    print(
        "CAVEAT : AUC with 8bit images might be lower than the floating point array (check <home>/pretrained/auc_roc*.npy)")
    for index in indices:
        if method_names[index] != 'CRFs' and method_names[index] != '2nd_manual':
            print("{} : {:04}".format(method_names[index], auc(fprs[index], tprs[index])))

    # plot results
    for index in indices:
        if method_names[index] == 'CRFs':
            plt.plot(fprs[index], tprs[index], colors[index] + '*', label=method_names[index].replace("_", " "))
        elif method_names[index] == '2nd_manual':
            plt.plot(fprs[index], tprs[index], colors[index] + '*', label='Human')
        else:
            plt.step(fprs[index], tprs[index], colors[index], where='post', label=method_names[index].replace("_", " "),
                     linewidth=1.5)

    # plot individual operation points
    for op_pt in op_pts:
        plt.plot(op_pt[0], op_pt[1], 'r.')

    plt.title('ROC Curve')
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.xlim(0, 0.3)
    plt.ylim(0.7, 1.0)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(fig_dir, "ROC.png"))
    plt.close()


def plot_AUC_PR(precisions, recalls, method_names, fig_dir, op_pts):
    # set font style
    font = {'family': 'serif'}
    matplotlib.rc('font', **font)

    # sort the order of plots manually for eye-pleasing plots
    colors = ['r', 'b', 'y', 'g', '#7e7e7e', 'm', 'c', 'k', '#cd919e'] if len(precisions) == 9 else ['r', 'y', 'm', 'g',
                                                                                                     'k']
    indices = [7, 2, 5, 3, 4, 6, 1, 8, 0] if len(precisions) == 9 else [4, 1, 2, 3, 0]

    # print auc
    print("****** Precision Recall AUC ******")
    print(
        "CAVEAT : AUC with 8bit images might be lower than the floating point array (check <home>/pretrained/auc_pr*.npy)")
    for index in indices:
        if method_names[index] != 'CRFs' and method_names[index] != '2nd_manual':
            print("{} : {:04}".format(method_names[index], auc(recalls[index], precisions[index])))

    # plot results
    for index in indices:
        if method_names[index] == 'CRFs':
            plt.plot(recalls[index], precisions[index], colors[index] + '*',
                     label=method_names[index].replace("_", " "))
        elif method_names[index] == '2nd_manual':
            plt.plot(recalls[index], precisions[index], colors[index] + '*', label='Human')
        else:
            plt.step(recalls[index], precisions[index], colors[index], where='post',
                     label=method_names[index].replace("_", " "), linewidth=1.5)

    # plot individual operation points
    for op_pt in op_pts:
        plt.plot(op_pt[0], op_pt[1], 'r.')

    plt.title('Precision Recall Curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0.5, 1.0)
    plt.ylim(0.5, 1.0)
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(fig_dir, "Precision_recall.png"))
    plt.close()


# 3D plot segmentation of liver and nodules
def plot_3d_seg(image, name, threshold=1, save=False):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    check = np.max(np.unique(image)) > 1
    verts, faces = measure.marching_cubes(image, threshold - 1)
    if check:
        verts2, faces2 = measure.marching_cubes(image, threshold)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.3)
    if check:
        mesh2 = Poly3DCollection(verts2[faces2], alpha=0.3)
    face_color = [1, 0.2, 0.2]
    if check:
        face_color2 = [0.3, 0.3, 1]
    mesh.set_facecolor(face_color)
    if check:
        mesh2.set_facecolor(face_color2)
    ax.add_collection3d(mesh)
    if check:
        ax.add_collection3d(mesh2)
    mesh_z = np.mean(image, axis=2)
    # mesh_y = np.mean(image,axis=1)
    # mesh_x = np.mean(image,axis=0)

    X = np.linspace(0, image.shape[0] - 1, image.shape[0])
    Y = np.linspace(0, image.shape[1] - 1, image.shape[1])
    Z = np.linspace(0, image.shape[2] - 1, image.shape[2])
    # a,b=np.meshgrid(Y,Z)
    c, d = np.meshgrid(X, Y)
    # e,f=np.meshgrid(X,Z)
    cest = ax.contourf(c, d, np.transpose(mesh_z), zdir='z', offset=0, cmap="Blues")
    # cest = ax.contourf(np.transpose(mesh_x),b,a,zdir='x', offset=0, cmap="Greys")
    # cest = ax.contourf(e,np.transpose(mesh_y),f,zdir="y", offset=image.shape[1], cmap="Greys")
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_ylim(0, image.shape[1])
    ax.set_xlim(0, image.shape[0])
    ax.set_zlim(0, image.shape[2])
    ax.set_title(name + ": 3D nodules and liver")
    if save:
        fig.savefig(
            "/home01/weileyi/jinqiangguo/jqg/py3EnvRoad/lung-segmentation-3d/" + name + "_3D_nodules_and_liver.png",
            bbox_inches='tight')
    plt.close(fig)
    del mesh, verts, faces, face_color
    if check:
        del mesh2, verts2, faces2, face_color2


# 3D Plot the complete image
def plot_3d_vol(image, name="Check", threshold=320, save=False):
    # Position the scan upright,
    p = image
    # so the head of the patient would be at the top facing the camera
    verts, faces = measure.marching_cubes(p, threshold - 1)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = '#0099ff'  # [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.view_init(30, 35)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    ax.set_title(name + "_3D_Volume_Scan")
    if save:
        fig.savefig("data/" + name + "_3D_Volume_Scan.png",
                    bbox_inches='tight')
    del mesh, verts, faces
    plt.close(fig)


def plot_3d_all(image, segm, name="Complete", threshold_bones=320, save=False):
    check = np.max(np.unique(segm)) > 1
    print("Finding marching cubes...")
    verts, faces = measure.marching_cubes(segm, 0)
    if check:
        verts2, faces2 = measure.marching_cubes(segm, 1)
    verts_vol, faces_vol = measure.marching_cubes(image, threshold_bones)
    fig = plt.figure(figsize=(15, 20))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    print("Computing polygons...")
    mesh = Poly3DCollection(verts[faces], alpha=0.4)
    if check:
        mesh2 = Poly3DCollection(verts2[faces2], alpha=0.7)
    mesh_vol = Poly3DCollection(verts_vol[faces_vol], alpha=0.25)
    print("Plotting...")
    face_color = [1, 0.2, 0.2]
    if check:
        face_color2 = [0.3, 0.3, 1]
    face_color_vol = [0, 0, 0]
    mesh.set_facecolor(face_color)
    if check:
        mesh2.set_facecolor(face_color2)
    mesh_vol.set_facecolor(face_color_vol)
    ax.add_collection3d(mesh)
    if check:
        ax.add_collection3d(mesh2)
    ax.add_collection3d(mesh_vol)
    mesh_z = np.mean(segm, axis=2)
    # mesh_y = np.mean(image,axis=1)
    # mesh_x = np.mean(image,axis=0)

    X = np.linspace(0, image.shape[0] - 1, image.shape[0])
    Y = np.linspace(0, image.shape[1] - 1, image.shape[1])
    Z = np.linspace(0, image.shape[2] - 1, image.shape[2])
    # a,b=np.meshgrid(Y,Z)
    c, d = np.meshgrid(X, Y)
    # e,f=np.meshgrid(X,Z)
    cest = ax.contourf(c, d, np.transpose(mesh_z), zdir='z', offset=0, cmap="Blues")
    # cest = ax.contourf(np.transpose(mesh_x),b,a,zdir='x', offset=0, cmap="Greys")
    # cest = ax.contourf(e,np.transpose(mesh_y),f,zdir="y", offset=image.shape[1], cmap="Greys")
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])
    ax.set_zlim(0, image.shape[2])
    ax.set_title(name + "_3D_Complete.png")
    if save:
        fig.savefig("data/" + name + "_3D_Complete.png",
                    bbox_inches='tight')
    plt.close("all")
    del mesh, mesh_vol, face_color, face_color_vol
    if check:
        del mesh2, face_color2, verts2, faces2
    del verts, verts_vol, faces, faces_vol
    gc.collect()


def make_hist(img, title, name, xfrom=-1200, xto=2000):
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(6, 4.5))
    plt.hist(img.ravel(), normed=0, bins=40, facecolor='blue')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Hounsfield units (HU)", fontsize=30)
    plt.ylabel("Frequency", fontsize=30)
    plt.xlim(xfrom, xto)
    plt.tight_layout()
    plt.title(title + " Windowing", fontsize=20)
    plt.savefig("data/" + name + "_Hist.png")
    plt.close()
    # plt.show()


# img = imgnii.get_data()
# mask = masknii.get_data()
# plot_3d_vol(img)
def draw_for_HU():
    df = pd.read_csv('Demo/idx-hu.csv')
    for i, item in df.iterrows():
        imgnii = nib.load(item[0])
        print('train data select nii file:' + item[0])
        niiName = item[0][5:]
        img = imgnii.get_data()
        # plot_3d_vol(img, name='Befor HU',threshold=40,save=True)
        make_hist(img, "Before", 'Before_HU_' + niiName, xfrom=-1200, xto=2000)
        sp = img.shape
        print(sp)
        # if not os.path.isdir("data/" + niiName + "/"):
        #     os.mkdir("data/" + niiName + "/")
        # for j in range(sp[-1]):
        #     cv2.imwrite("data/" + niiName + "/" + niiName + "_z_before_hu_" + str(j) + ".png", img[:, :, j])
        # for j in range(sp[-2]):
        #     cv2.imwrite("data/" + niiName + "/" + niiName + "_y_before_hu_" + str(j) + ".png", img[:, j, :])
        # for j in range(sp[0]):
        #     cv2.imwrite("data/" + niiName + "/" + niiName + "_x_before_hu_" + str(j) + ".png", img[j, :, :])
        img = np.clip(img, -100, 200)
        img = img.astype(np.float32)
        make_hist(img, "After", 'After_HU_' + niiName, xfrom=-120, xto=250)
        # for j in range(sp[-1]):
        #     cv2.imwrite("data/" + niiName + "/" + niiName + "_z_after_hu_" + str(j) + ".png", img[:, :, j])
        # for j in range(sp[-2]):
        #     cv2.imwrite("data/" + niiName + "/" + niiName + "_y_after_hu_" + str(j) + ".png", img[:, j, :])
        # for j in range(sp[0]):
        #     cv2.imwrite("data/" + niiName + "/" + niiName + "_x_after_hu_" + str(j) + ".png", img[j, :, :])
        # nib.save(nib.Nifti1Image(img.astype('float'), affine=imgnii.get_affine()),
        #          "data/" + niiName[:len(niiName) - 4] + '.nii.gz')


def plotNNFilterOverlay(input_im, units, figure_id, interp='bilinear',
                        colormap=cm.jet, colormap_lim=None, title='', alpha=0.5, row=2):
    plt.ion()
    filters = units.shape[2]
    f, ax = plt.subplots(row, filters // row)
    for i in range(row):
        for j in range(filters // row):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].imshow(input_im[:, :, i * filters // row + j], interpolation=interp, cmap='gray')
            ax[i, j].imshow(units[:, :, i * filters // row + j], interpolation=interp, cmap=colormap, alpha=alpha)
            if colormap_lim:
                ax[i, j].clim(colormap_lim[0], colormap_lim[1])

    # for i in range(filters):
    #     plt.imshow(input_im[:, :, i], interpolation=interp, cmap='gray')
    #     plt.imshow(units[:, :, i], interpolation=interp, cmap=colormap, alpha=alpha)
    #     plt.axis('off')
    #     plt.title(title, fontsize='small')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    # plt.tight_layout(0.01,0.001)
    plt.savefig(title, format='png', transparent=True)


def plot_feature_map(img_slice, L, imgname, row=4):
    # ======pic one

    image = Image.fromarray(img_slice[:, :, 0])
    image = image.resize((L[0].shape[0], L[0].shape[1]))
    image = np.expand_dims(np.array(image), -1)
    plotNNFilterOverlay(np.tile(image, len(L)), np.transpose(L, (1, 2, 0)), 1, title=imgname + '-1.png',
                        row=row)

    # ======pic two
    f, ax = plt.subplots(row, len(L) // row)
    for i in range(row):
        for j in range(len(L) // row):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].imshow(L[i * len(L) // row + j], interpolation='nearest', cmap='jet')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.savefig(imgname + '-2.png', format='png', transparent=True)


def plot_feature_map_entrance(dir='../vis/3dircad/'):
    image_label = np.load(dir + 'img-grads-final-0.npy')
    image_file = np.load(dir + 'img-raw-0.npy')
    imgla = np.load(dir + 'msk-0.npy')
    img_slice = np.load(dir + 'img-raw-slice-0.npy')[0]
    from keras.preprocessing.image import array_to_img
    im = array_to_img(img_slice)
    im.save(dir + 'raw_img.png')
    # =======================================================
    L = np.load(dir + 'img-atten4-0.npy')
    plot_feature_map(img_slice, L, dir + "img_atten4_layer", row=4)
    # # =======================================================
    L = np.load(dir + 'img-atten3-0.npy')
    plot_feature_map(img_slice, L, dir + "img_atten3_layer", row=4)
    # # =======================================================
    L = np.load(dir + 'img-atten2-0.npy')
    plot_feature_map(img_slice, L, dir + "img_atten2_layer", row=8)
    # =======================================================
    tp = np.transpose(np.nonzero(imgla[:, :, :]))
    minx, miny, minz = np.min(tp, axis=0)
    maxx, maxy, maxz = np.max(tp, axis=0)
    num_vis = 8
    img_vis = np.zeros((image_file.shape[1], image_file.shape[2], num_vis))
    feature_vis = np.zeros((image_file.shape[1], image_file.shape[2], num_vis))
    for i in range(num_vis):
        x = minz + i * (maxz - minz) // (num_vis + 1)
        img_vis[:, :, i] = image_file[x, :, :, 0] * 255.
        feature_vis[:, :, i] = image_label[x, :, :, 0]
    plotNNFilterOverlay(img_vis, feature_vis, 1, title=dir + "img_final_layer.png", row=2)


def plot_probability_map(dir='../vis/3dircad/'):
    import nibabel as nib
    img_soft = nib.load(
        'F:\\xunleixiazai\\xftp_tmp\\3DIRCADb\\RA-UNet\\predciton_final\\soft\\test-segmentation-0.nii.gz').get_data()
    image_gt = np.load(dir + 'msk-0.npy')
    image_file = np.load(dir + 'img-0.npy')
    tp = np.transpose(np.nonzero(image_gt[:, :, :]))
    minx, miny, minz = np.min(tp, axis=0)
    maxx, maxy, maxz = np.max(tp, axis=0)
    num_vis = 8
    img_vis = np.zeros((image_file.shape[0], image_file.shape[1], num_vis))
    feature_vis = np.zeros((image_file.shape[0], image_file.shape[1], num_vis))
    for i in range(num_vis):
        x = minz + i * (maxz - minz) // (num_vis + 1)
        img_vis[:, :, i] = image_file[:, :, x] * 255.
        feature_vis[:, :, i] = img_soft[:, :, x]

    plotNNFilterOverlay(img_vis, feature_vis, 1, title=dir + "img_final_3d_layer.png", row=2)


draw_for_HU()
