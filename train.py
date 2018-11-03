from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback, LambdaCallback, \
    CSVLogger
from keras.utils import multi_gpu_model
import os
import time

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from keras.utils.vis_utils import plot_model
from keras.optimizers import *
from build_model import get_net
from preprocess import *
from utils.utils import *
from keras.models import model_from_json
import json

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

CFG = configure.cfg
LOG_FILE = CFG['log_file']
log_path_experiment = ''
log_tensorboard_filepath = ''
logger = ''


def get_log_path():
    return log_path_experiment


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
                                                      mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)


# because the keras bug. if para we must use the origion model to save the shared weights
class ModelCallBackForMultiGPU(Callback):
    def __init__(self, model, log_path_experiment):
        self.model_to_save = model
        self.log_path = log_path_experiment

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 1 == 0:
            self.model_to_save.save_weights(
                self.log_path + '/model_at_' + time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(
                    time.time())) + '_epoch_%05d.hdf5' % epoch)
            # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


def train_liver_slice(algorithm='unet', patch_data_dir='data/slices/slices_filtered_0.66', val_ratio=0.289,
                      init_lr=1e-3, gpus=2, n_rounds=60, batch_size=90):
    # 'unet  vgg_fcn'
    log_path_experiment = LOG_FILE + '/' + algorithm
    define_log(LOG_DIR, algorithm)
    # 70 for 3gpu
    model_out_dir = "{}/model_{}".format(log_path_experiment, val_ratio)
    if not os.path.isdir(model_out_dir):
        os.makedirs(model_out_dir)
    train_batch_fetcher = TrainBatchFetcher(patch_data_dir, val_ratio)
    val_imgs, val_labels = train_batch_fetcher.vali_data()
    print(val_imgs.shape)
    print(val_labels.shape)
    # Build model
    model = get_net(val_imgs[0].shape, algorithm)
    with open(os.path.join(model_out_dir, "g_{}.json".format(val_ratio)), 'w') as f:
        f.write(model.to_json())
    parallel_model = multi_gpu_model(model, gpus=gpus)
    # parallel_model = model
    adam = Adam(lr=init_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # parallel_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    parallel_model.compile(optimizer=adam, loss=dice_coef_loss, metrics=['accuracy', dice_coef])
    checkpointer = ModelCallBackForMultiGPU(model, log_path_experiment)
    csv_logger = CSVLogger(log_path_experiment + '/training.csv', append=True)
    reduce = ReduceLROnPlateau(factor=0.1, patience=20, verbose=2)
    cbks = [checkpointer, reduce, csv_logger]
    # Visualize model
    plot_model(model, to_file=log_path_experiment + '/model.pdf', show_shapes=True)

    model.summary()
    ##########################################################################################
    # Train the model
    parallel_model.fit_generator(generator=train_batch_fetcher.next(batch_size),
                                 steps_per_epoch=train_batch_fetcher.num_training // batch_size,
                                 epochs=n_rounds,
                                 verbose=0,
                                 callbacks=cbks,
                                 validation_data=(val_imgs, val_labels),
                                 validation_steps=train_batch_fetcher.num_validation // batch_size)

    # Evaluate the model
    score = parallel_model.evaluate(
        val_imgs, val_labels,
        batch_size=batch_size, verbose=2
    )
    print('**********************************************')
    print('Test score:', score)


def train_liver_patches_with_alg(algorithm='resunet', patch_data_dir='data/patches', val_ratio=0.2,
                                 pre_trained_weight=None,
                                 model_path=None, init_lr=1e-3, batch_size=60, n_rounds=200, gpu=2, iskfold=False, k=5,
                                 patience=20):
    # algorithm = '3dunet'
    log_path_experiment = LOG_FILE + '/' + algorithm
    define_log(LOG_DIR, algorithm)
    # 7-8 per gpu  21 64*64  3gpu;;;;        for attention res-unet 3 per gpu
    model_out_dir = "{}/model_{}".format(log_path_experiment, val_ratio)
    if not os.path.isdir(model_out_dir):
        os.makedirs(model_out_dir)
    if iskfold:
        cvscores = []
        for i in range(0, k):
            print(str(i) + 'th fold training')
            train_batch_fetcher = TrainBatchFetcher(patch_data_dir, 1. / k * (i + 1), valid_from=1. / k * i)
            val_imgs, val_labels = train_batch_fetcher.get_one()
            model = get_net(val_imgs[0].shape, algorithm)
            with open(os.path.join(model_out_dir, "g_{}.json".format(val_ratio)), 'w') as f:
                f.write(model.to_json())
            plot_model(model, to_file=log_path_experiment + '/model.pdf', show_shapes=True)
            checkpointer = ModelCallBackForMultiGPU(model, log_path_experiment)
            reduce = ReduceLROnPlateau(factor=0.1, patience=patience, verbose=2)
            json_log = open(log_path_experiment + '/val_loss_log.json', mode='at', buffering=1)
            csv_logger = CSVLogger(log_path_experiment + '/training.csv', append=True)
            cbks = [checkpointer, reduce, csv_logger]
            adam = Adam(lr=init_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199)
            # parallel_model = multi_gpu_model(model, gpus=gpu)
            parallel_model = model
            # if val_imgs[0].shape[0] != 224:
            #     parallel_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
            # else:
            parallel_model.compile(optimizer=adam, loss=dice_coef_loss, metrics=['accuracy', dice_coef])
            parallel_model.fit_generator(generator=train_batch_fetcher.next(batch_size),
                                         steps_per_epoch=train_batch_fetcher.num_training // batch_size,
                                         epochs=n_rounds,
                                         verbose=2,
                                         callbacks=cbks,
                                         validation_data=train_batch_fetcher.next(batch_size, 'vali'),
                                         validation_steps=train_batch_fetcher.num_validation // batch_size)

            # evaluate the model
            # scores = parallel_model.evaluate(val_imgs, val_labels, verbose=2)
            # print("%s: %.2f%%" % (parallel_model.metrics_names[1], scores[1] * 100))
            # del parallel_model, model
            # gc.collect()
            # cvscores.append(scores[1] * 100)
        # print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    else:
        train_batch_fetcher = TrainBatchFetcher(patch_data_dir, val_ratio)
        val_imgs, val_labels = train_batch_fetcher.get_one()
        # with tf.device('/cpu:0'):
        if model_path == None:
            model = get_net(val_imgs[0].shape, algorithm)
            with open(os.path.join(model_out_dir, "g_{}.json".format(val_ratio)), 'w') as f:
                f.write(model.to_json())
            model.summary()
        else:
            with open(model_path, 'r') as f:
                model = model_from_json(f.read())
            model.load_weights(pre_trained_weight)
            print('load pretrained_weight')
        checkpointer = ModelCallBackForMultiGPU(model, log_path_experiment)
        # checkpointer = ModelCheckpoint(filepath=log_path_experiment+"/checkpoint-{epoch:02d}e- val_acc_{val_acc: .4f}.hdf5", save_best_only=False, verbose=1,
        #                 period=1)
        json_log = open(log_path_experiment + '/val_loss_log.json', mode='at', buffering=1)
        json_logging_callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs: json_log.write(
                json.dumps({'epoch': epoch, 'val_loss': logs['val_loss'],
                            'val_acc': logs['val_acc'],
                            'val_dice_coef': logs['val_dice_coef']}) + '\n')
            # on_train_end=lambda logs: json_log.close()
        )
        reduce = ReduceLROnPlateau(factor=0.1, verbose=1, patience=20)
        csv_logger = CSVLogger(log_path_experiment + '/training.csv', append=True)
        cbks = [checkpointer, reduce, csv_logger]
        plot_model(model, to_file=log_path_experiment + '/model.pdf', show_shapes=True)
        parallel_model = multi_gpu_model(model, gpus=gpu)
        # parallel_model = model
        # adam = Adam(lr=init_lr)
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # if val_imgs[0].shape[0] != 224:
        #     parallel_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        # else:
        # parallel_model.compile(optimizer=adam, loss=dice_coef_loss, metrics=['accuracy', dice_coef])
        # parallel_model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199),
        #               loss='binary_crossentropy', metrics=['accuracy'])
        parallel_model.compile(optimizer=Adam(lr=init_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199),
                               loss=dice_coef_loss, metrics=['accuracy', dice_coef])

        parallel_model.fit_generator(generator=train_batch_fetcher.next(batch_size),
                                     steps_per_epoch=train_batch_fetcher.num_training // batch_size,
                                     epochs=n_rounds,
                                     verbose=2,
                                     callbacks=cbks,
                                     validation_data=train_batch_fetcher.next(batch_size, 'vali'),
                                     validation_steps=train_batch_fetcher.num_validation // batch_size
                                     )
        print('**********************************************')


if __name__ == '__main__':
    # train slices raunet1
    # train_liver_slice(algorithm='liver_att_resunet_2d', val_ratio=0.2,
    #                   patch_data_dir='data/patches/liver_slices', n_rounds=100)
    # train slices raunet2
    # train_liver_patches_with_alg(algorithm='liver_att_resunet_3d',
    #                              patch_data_dir='data/patches/liver_224',
    #                              gpu=2, init_lr=1e-3, batch_size=2, n_rounds=50, iskfold=True, patience=15)

    train_liver_patches_with_alg(algorithm='liver_tumor_att_resunet_3d',
                                 patch_data_dir='data/patches/liver_tumor_128_350',
                                 gpu=2, init_lr=1e-3, batch_size=6, n_rounds=60, iskfold=True, patience=15)
