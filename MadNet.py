import keras
import tensorflow as tf
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Lambda, Dropout
from keras.layers import AveragePooling2D, Input, Flatten
from keras.layers import LeakyReLU, Dot
from keras.regularizers import l2
from keras.models import Model
import keras.backend as K
from keras.backend import concatenate
import numpy as np

def resnet_layer_siamese(input1, input2,
                         num_filters=16,
                         kernel_size=3,
                         strides=1,
                         activation='relu',
                         batch_normalization=True,
                         conv_first=True, weight_decay=0.0002, conv_layer_list=[]):
    '''
    a single block of resnet duplicated so as to create a Siamese Network
    :param input1:
    :param input2:
    :param num_filters:
    :param kernel_size:
    :param strides:
    :param activation:
    :param batch_normalization:
    :param conv_first:
    :param weight_decay:
    :param conv_layer_list:
    :return:
    '''
    n = kernel_size * kernel_size * num_filters
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  use_bias=False,
                  kernel_initializer=tf.random_normal_initializer(
                      stddev=np.sqrt(2.0 / n)),
                  kernel_regularizer=l2(weight_decay))
    x1 = input1
    x2 = input2

    if conv_first:
        x1 = conv(x1)
        x2 = conv(x2)
        if batch_normalization:
            bn = BatchNormalization(momentum=0.9)
            x1 = bn(x1)
            x2 = bn(x2)

        if activation is not None:
            if activation == 'leaky-relu':
                x1 = LeakyReLU(alpha=0.1)(x1)
                x2 = LeakyReLU(alpha=0.1)(x2)

            else:
                x1 = Activation(activation)(x1)
                x2 = Activation(activation)(x2)

    else:
        if batch_normalization:
            bn = BatchNormalization(momentum=0.9)

            x1 = bn(x1)
            x2 = bn(x2)

        if activation is not None:

            if activation == 'leaky-relu':
                x1 = LeakyReLU(alpha=0.1)(x1)
                x2 = LeakyReLU(alpha=0.1)(x2)
            else:
                x1 = Activation(activation)(x1)
                x2 = Activation(activation)(x2)
        x1 = conv(x1)
        x2 = conv(x2)

    conv_layer_list.append([Dot(1, normalize=True)([Flatten()(x1), Flatten()(x1)]), Flatten()(x1),
                            Flatten()(x1)])  # adding the siamese and two embedding for reducing the variance

    return x1, x2, conv_layer_list


def pad_depth(x, desired_channels):
    y = K.zeros_like(x)
    new_channels = desired_channels - x.shape.as_list()[-1]
    y = y[:, :, :, :new_channels]
    return concatenate([x, y])


def resnet_v1_siamese(input_shape,
                      depth,
                      num_classes=10,
                      weight_decay=0.0,
                      embedding_activation='leaky-relu',
                      embedding_aux_loss='cosine',
                      reduce_variance=False,
                      reduce_jacobian_loss=False,
                      load_weights='',
                      reduce_juccobian_coeff=0.01):
    '''
    A resent model with the different MAD loss components
    :param input_shape:
    :param depth: numst be 6n+2 (32,56...)
    :param num_classes: number of classes
    :param weight_decay: decay to use for an l2 regularization
    :param embedding_activation: replace embedding layer activation functions
    :param embedding_aux_loss: loss to use for Siamese currently supporting reduce of margin and cosine distance
    :param reduce_variance: bool, if True adding the reduce variance loss
    :param reduce_jacobian_loss: bool, if True adding the reduce Jacobian loss
    :param load_weights: path to pretrained model to load, must be with identical configuration
    :param reduce_juccobian_coeff
    :return:
    '''

    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    in1 = Input(shape=input_shape)
    in2 = Input(shape=input_shape)
    conv_layer_list = []

    x1, x2, conv_layer_list = resnet_layer_siamese(in1, in2, activation=embedding_activation, conv_first=True,
                                                   weight_decay=weight_decay, conv_layer_list=conv_layer_list)
    # Instantiate the stack of residual units
    first_iter = True
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            if not first_iter:

                bn = BatchNormalization(momentum=0.9)
                if embedding_activation == 'leaky-relu':

                    lr = LeakyReLU(alpha=0.1)
                else:
                    lr = Activation(embedding_activation)
                y1 = bn(x1)
                y1 = lr(y1)
                y2 = bn(x2)
                y2 = lr(y2)
            else:
                y1 = x1
                y2 = x2
            y1, y2, conv_layer_list = resnet_layer_siamese(y1, y2,
                                                           num_filters=num_filters,
                                                           strides=strides,
                                                           activation=embedding_activation, conv_first=True,
                                                           weight_decay=weight_decay, conv_layer_list=conv_layer_list)
            y1, y2, conv_layer_list = resnet_layer_siamese(y1, y2,
                                                           num_filters=num_filters,
                                                           activation=None, conv_first=True, weight_decay=weight_decay,
                                                           batch_normalization=False, conv_layer_list=conv_layer_list)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                ap = AveragePooling2D(2, 2, 'valid')
                x1 = ap(x1)
                x2 = ap(x2)

                x1 = Lambda(pad_depth, arguments={'desired_channels': y1.shape[-1]})(x1)
                x2 = Lambda(pad_depth, arguments={'desired_channels': y2.shape[-1]})(x2)

            x1 = keras.layers.add([x1, y1])
            x2 = keras.layers.add([x2, y2])

            first_iter = False

        num_filters *= 2

    bn = BatchNormalization(momentum=0.9)
    x1 = bn(x1)
    x2 = bn(x2)
    if embedding_activation == 'leaky-relu':
        lr = LeakyReLU(alpha=0.1)
    elif embedding_activation == 'tanh':
        print("Using tanh activation")
        lr = Activation('tanh')

    x1 = lr(x1)
    x2 = lr(x2)
    ap = AveragePooling2D(pool_size=int(input_shape[0] / 4), name='bottleneck')
    x1 = ap(x1)
    x2 = ap(x2)

    emb1 = Flatten()(x1)
    emb2 = Flatten()(x2)

    dense = Dense(num_classes, name='logits',
                  kernel_initializer=tf.uniform_unit_scaling_initializer(factor=1.0),
                  kernel_regularizer=l2(weight_decay),
                  bias_initializer=tf.constant_initializer())


    logits1 = dense(emb1)
    logits2 = dense(emb2)

    output1 = Activation('softmax', name='main_output1')(logits1)
    output2 = Activation('softmax', name='main_output2')(logits2)
    if embedding_aux_loss == 'cosine':
        aux_out = Dot(1, normalize=True)([emb1, emb2])
    elif embedding_aux_loss == 'margin':
        print("using margin loss")
        aux_out = Lambda(
            lambda l: K.concatenate((K.expand_dims(l[0], axis=-1), K.expand_dims(l[1], axis=-1)), axis=-1))(
            [emb1, emb2])

    output_list = [output1, output2, aux_out]
    if reduce_variance:
        output_list += [emb1, emb2]
    if reduce_jacobian_loss:
        jacobian_output1 = Lambda(lambda l: reduce_juccobian_coeff * K.sqrt(
            K.sum(K.pow(K.gradients(output1, l)[0], 2), axis=(1, 2, 3))), output_shape=[1])(in1)
        jacobian_output2 = Lambda(lambda l: reduce_juccobian_coeff * K.sqrt(
            K.sum(K.pow(K.gradients(output2, l)[0], 2), axis=(1, 2, 3))), output_shape=[1])(in2)

        output_list += [jacobian_output1, jacobian_output2]

    model = Model(inputs=[in1, in2], outputs=output_list)

    if load_weights != '': #loading weights
        temp_model = keras.Model(model.input[0],model.get_layer('main_output1').output)
        temp_model.load_weights(load_weights)
        temp_model = keras.Model(model.input[1],model.get_layer('main_output2').output)
        temp_model.load_weights(load_weights)

    return model



def create_madnet_resnet(image_shape,
                       depth,
                       num_classes,weight_decay=0.0,
                         embedding_activation='leaky-relu',
                         embedding_aux_loss='cosine',
                         reduce_variance=False,
                         reduce_jacobian_loss=False,
                         load_weights='',
                         reduce_juccobian_coeff=0.01):
    '''
    creates MadNet, see resnet_v1_siamese for parameters
    :param image_shape:
    :param depth:
    :param num_classes:
    :param weight_decay:
    :param embedding_activation:
    :param embedding_aux_loss:
    :param reduce_variance:
    :param reduce_jacobian_loss:
    :param load_weights:
    :param reduce_juccobian_coeff:
    :return:
    '''

    model = resnet_v1_siamese(input_shape=image_shape,
                              depth=depth,
                              num_classes=num_classes,
                              weight_decay=weight_decay,
                              embedding_activation=embedding_activation,
                              embedding_aux_loss=embedding_aux_loss,
                              reduce_variance=reduce_variance,
                              reduce_jacobian_loss=reduce_jacobian_loss,
                              load_weights=load_weights,
                              reduce_juccobian_coeff=reduce_juccobian_coeff)
    if load_weights != '':
        optimizer = keras.optimizers.SGD(1e-3, momentum=0.9)
    else:
        optimizer = keras.optimizers.SGD(0.1, momentum=0.9)

    def reduce_variance_loss(labels, embeddings):
        ''' loss for reducing variance per class '''
        loss = 0
        sq_labels = K.squeeze(labels, 1)
        for cls in range(num_classes):
            # extracting per class embeddings
            ind_class = K.tf.where(K.equal(sq_labels, cls))
            is_empty = tf.equal(tf.size(ind_class), 0)
            class_embeddings = K.switch(is_empty, K.variable([0, 0]), K.squeeze(K.gather(embeddings, ind_class), 1))
            mu = K.mean(class_embeddings, axis=0)
            sigma = K.mean(K.square(class_embeddings - mu), axis=0)
            # demanding zero within class variance
            loss += K.mean(K.square(sigma))

        return loss

    loss_list = ['categorical_crossentropy', 'categorical_crossentropy', 'mean_absolute_error']

    if reduce_variance:
        loss_list.append(reduce_variance_loss)
        loss_list.append(reduce_variance_loss)

    if reduce_jacobian_loss:
        loss_list += ['mean_squared_error', 'mean_squared_error']

    model.compile(loss=loss_list,
                      optimizer=optimizer,
                      metrics=['accuracy'])
    model.summary()
    return model