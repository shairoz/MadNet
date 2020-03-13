import keras
import tensorflow as tf
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import argparse
import os
import random
from MadNet_utils import load_cifar10_05,load_mnist_05
from MadNet import create_madnet_resnet
import numpy as np


def train_model(data_loader,
                depth,
                log_path,
                weight_decay=0,
                embedding_activation='leaky-relu',
                embedding_aux_loss='cosine',
                reduce_variance=False,
                dataset='cifar10',
                reduce_jacobian_loss=False,
                label_smoothing=0,
                load_weights='',
                batch_size=128):
    '''

    :param data_loader: see utils
    :param depth: depth of resnet must be 6n+2 (32,56...)
    :param log_path: path to write model
    :param weight_decay:
    :param embedding_activation:
    :param embedding_aux_loss:
    :param reduce_variance:
    :param dataset:
    :param reduce_jacobian_loss:
    :param label_smoothing:
    :param load_weights:
    :param batch_size:
    :return:
    '''


    (x_train, y_train), (x_test, y_test) = data_loader
    num_classes = int(np.max(y_train) + 1)

    model_type = 'resnt'
    model = create_madnet_resnet(x_test.shape[1:],
                                 depth,
                                 num_classes,
                                 weight_decay=weight_decay,
                                 embedding_activation=embedding_activation,
                                 embedding_aux_loss=embedding_aux_loss,
                                 reduce_variance=reduce_variance,
                                 reduce_jacobian_loss=reduce_jacobian_loss,
                                 load_weights=load_weights,
                                 reduce_juccobian_coeff = 0.01)


    model_text_addition = ''

    if embedding_activation != 'leaky-relu':
        model_text_addition = model_text_addition + '_' + embedding_activation
    if embedding_aux_loss != 'cosine':
        model_text_addition = model_text_addition + '_' + embedding_aux_loss
    if reduce_variance:
        model_text_addition += 'reduce_variance'
    if reduce_jacobian_loss:
        model_text_addition += '_reduce_jacobian_loss_' + str()
    if label_smoothing > 0:
        model_text_addition += '_label_smoothing_' + str(label_smoothing)
    if load_weights != '':
        model_text_addition += '_fine_tuned'

    y_train_categorical = to_categorical(y_train, num_classes)
    y_test_categorical = to_categorical(y_test, num_classes)
    if label_smoothing > 0:
        for ind, y in enumerate(y_test_categorical):
            y_test_categorical[ind] = (1 - label_smoothing) / (num_classes - 1)
            y_test_categorical[ind][y_test[ind]] = label_smoothing
        for ind, y in enumerate(y_train_categorical):
            y_train_categorical[ind] = (1 - label_smoothing) / (num_classes - 1)
            y_train_categorical[ind][y_train[ind]] = label_smoothing

    def random_crop(img):
        # Note: image_data_format is 'channel_last'
        assert img.shape[2] == 3 or img.shape[2] == 1
        # img = np.pad(img,[2,2,1],mode='constant',constant_values=0)
        pad = np.ones((img.shape[0] + 4, img.shape[1] + 4, img.shape[2])) * (
            -0.5)  # padding is done after image was scaled to 0.5,-0.5
        pad[2:img.shape[0] + 2, 2:img.shape[1] + 2, :] = img
        img = pad
        height, width = img.shape[0], img.shape[1]
        if dataset == 'cifar10':
            dy, dx = 32, 32
        elif dataset == 'mnist':
            dy, dx = 28, 28
        elif dataset == 'tiny_imagenet':
            dy, dx = 64, 64
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y + dy), x:(x + dx), :]

    def lr_schedule(epoch):
        if dataset == 'cifar10':
            if epoch < 40000 / (50000 / batch_size):
                lr = 0.1
            elif epoch < 70000 / (50000 / batch_size):
                lr = 0.01
            elif epoch < 100000 / (50000 / batch_size):
                lr = 0.001
            elif epoch < 130000 / (50000 / batch_size):
                lr = 0.0001
            else:
                lr = 0.00001
        else:  # mnist
            if epoch < 7500 / (50000 / batch_size):
                lr = 0.1
            elif epoch < 15000 / (50000 / batch_size):
                lr = 0.01
            elif epoch < 22500 / (50000 / batch_size):
                lr = 0.001
            elif epoch < 25000 / (50000 / batch_size):
                lr = 0.0001
            else:
                lr = 0.00001

        print('Learning rate: ', lr)
        tf.summary.scalar('learning_rate', lr)
        return lr

    train_class_inds = []
    for i in range(num_classes):
        train_class_inds.append(np.where(np.argmax(y_train_categorical, axis=1) == i)[0])
    test_class_inds = []
    for i in range(num_classes):
        test_class_inds.append(np.where(np.argmax(y_test_categorical, axis=1) == i)[0])

    def generator(x, y, mode='train'):
        if mode == 'train':
            class_inds = train_class_inds
        else:
            class_inds = test_class_inds
        if mode == 'train':
            while True:
                ind1 = np.random.choice(len(x), batch_size)
                epoch_classes = np.asarray(range(0, num_classes))
                x1 = x[ind1]
                y1 = y[ind1]
                ind2 = []
                p_flip1 = np.random.uniform(0, 1, batch_size)
                p_flip2 = np.random.uniform(0, 1, batch_size)
                p_pair = np.random.uniform(0, 1, batch_size)
                cosine_sim = np.zeros(batch_size)
                for i in range(batch_size):
                    if p_pair[i] > 0.5:  # choosing from same class
                        ind2.append(np.random.choice(class_inds[np.argmax(y1[i])]))
                        cosine_sim[i] = 1
                    else:  # different class
                        # class_list= list(range(num_classes))
                        class_list = list(epoch_classes)
                        class_list.remove(np.argmax(y1[i]))
                        ind2.append(np.random.choice(class_inds[random.choice(class_list)]))
                        if embedding_activation == 'tanh':
                            cosine_sim[i] = -1
                        else:
                            cosine_sim[i] = 0
                x2 = x[ind2]
                y2 = y[ind2]

                for i in range(batch_size):
                    if p_flip1[i] > 0.5:
                        x1[i] = np.fliplr(x1[i])
                    if p_flip2[i] > 0.5:
                        x2[i] = np.fliplr(x2[i])

                    x1[i] = random_crop(x1[i])
                    x2[i] = random_crop(x2[i])
                # using transformations for classification and cosine

                cosine_label = np.asarray(cosine_sim)

                if embedding_aux_loss == 'margin':
                    cosine_label = np.repeat(np.expand_dims(cosine_label, -1), 64, axis=-1)
                    cosine_label = np.repeat(np.expand_dims(cosine_label, -1), 2, axis=-1)

                output_label = [y1, y2]
                if reduce_variance:
                    output_label += [cosine_label, np.expand_dims(np.argmax(y1, axis=1), axis=-1),
                                         np.expand_dims(np.argmax(y2, axis=1), axis=-1)]
                else:
                    output_label += [cosine_label]
                if reduce_jacobian_loss:
                    output_label += [np.zeros(batch_size), np.zeros(batch_size)]
                yield [x1, x2], output_label
        else:  # test
            while True:
                for b in range(int(len(x) / batch_size)):
                    x1 = x[b * batch_size:(b + 1) * batch_size]
                    y1 = y[b * batch_size:(b + 1) * batch_size]

                    output_label = [y1, y1]
                    if reduce_variance:
                        output_label += [np.ones(batch_size), np.expand_dims(np.argmax(y1, axis=1), axis=-1),
                                             np.expand_dims(np.argmax(y1, axis=1), axis=-1)]
                    else:
                        output_label += [np.ones(batch_size)]
                    if reduce_jacobian_loss:
                        output_label += [np.zeros(batch_size), np.zeros(batch_size)]

                    yield [x1, x1], output_label

    path = os.path.join(log_path, model_type + '_siamese_' + str(depth) + model_text_addition + '_weight_decay_' + str(
        weight_decay) + '_' + dataset + '.hdf')

    check_point = ModelCheckpoint(path, save_best_only=True, monitor='val_main_output1_acc', save_weights_only=True)
    if dataset == 'cifar10':
        epochs = int(160000 / (len(x_train) / batch_size))  # Performing fixed  steps
    else:  # mnist
        epochs = int(30000 / (len(x_train) / batch_size))  # Performing fixed  steps

    tbCallBack = keras.callbacks.TensorBoard(
        log_dir=os.path.join(log_path, 'graph_' + model_text_addition + str(weight_decay)),
        histogram_freq=0,
        write_graph=True, write_images=False)

    if load_weights != '':
        if dataset == 'cifar10':
            epochs = 50
        elif dataset =='mnist':
            epochs = 20
        lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=int(epochs/5))
    else:
        lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)


    model.fit_generator(generator(x_train, y_train_categorical, mode='train'),
                            steps_per_epoch=int(len(x_train) / batch_size),
                            validation_data=generator(x_test, y_test_categorical, mode='test'),
                            validation_steps=int(len(x_test) / batch_size),
                            epochs=epochs, verbose=1, workers=1,
                            callbacks=[check_point, tbCallBack, lr_callback])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-log_path', '--log_path', help='path to write model checkpoint', required=True)
    parser.add_argument('-weight_decay', '--weight_decay', help='weight decay', required=False, type=float,
                        default=0.0002)
    parser.add_argument('-resnet_depth', '--resnet_depth', help='resnet depth, must be 6n+2, e.g., 32,56',
                        required=True, type=int)
    parser.add_argument('-embedding_activation', '--embedding_activation', help='embedding_activation', required=False,
                        default='leaky-relu', type=str)
    parser.add_argument('-embedding_aux_loss', '--embedding_aux_loss', help='embedding_aux_loss', required=False,
                        default='cosine', type=str)
    parser.add_argument('-reduce_variance', '--reduce_variance',
                        help='adding the variance reduction to the loss function',
                        required=False,
                        action='store_true',
                        default=False)
    parser.add_argument('-reduce_jacobian_loss', '--reduce_jacobian_loss',
                        help='adding the Jacobian reduction to the loss function',
                        required=False,
                        action='store_true',
                        default=False)
    parser.add_argument('-dataset', '--dataset', help='dataset', required=False)
    parser.add_argument('-label_smoothing', '--label_smoothing', help='label_smoothing', required=False, type=float,
                        default=0)
    parser.add_argument('-load_weights', '--load_weights', help='start with pretrained model', required=False, type=str,
                        default='')

    args = vars(parser.parse_args())
    if args['dataset'] == 'cifar10':
        data_loader = load_cifar10_05()
    elif args['dataset'] == 'mnist':
        data_loader = load_mnist_05()

    train_model(data_loader,
                args['resnet_depth'],
                args['log_path'],
                weight_decay=args['weight_decay'],
                embedding_activation=args['embedding_activation'],
                embedding_aux_loss=args['embedding_aux_loss'],
                reduce_variance=args['reduce_variance'],
                dataset=args['dataset'],
                reduce_jacobian_loss=args['reduce_jacobian_loss'],
                label_smoothing=args['label_smoothing'],
                load_weights=args['load_weights'])

