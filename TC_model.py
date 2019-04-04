from keras.models import Model
from keras.layers import Input, concatenate, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import Reshape, Activation, Dropout, BatchNormalization
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D
from keras.optimizers import *
import keras

from TC_data import *

def get_unet_2D(img_rows, img_cols, sgd, dropoutpct, dropout=True, activation='relu'):
    """
    Initialisation of Network, which consists of a convulutional Unet for 2D images.
    
    Parameters:
        img_rows: int, size x-dimension
        img_cols: int, size y-dimension
        sgd: optimizer of loss funtion
        dropoutpct: float, amount of dropuot
        dropout: boolian that determines if dropout is applied or not
        activation: string, describes the activation function
            
    Returns:
        Unet_2D: initialised network
    """
    inputs = Input(shape=(img_rows, img_cols, 1)) #(1, dim1, dim2, channels)
    # print(inputs.shape)

    conv1 = Convolution2D(64, (3, 3), padding='same', data_format='channels_last')(inputs)
    conv1 = Activation(activation)(conv1)
    if dropout: conv1 = Dropout(dropoutpct)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Convolution2D(64, (3, 3), padding='same', data_format='channels_last')(conv1)
    conv1 = Activation(activation)(conv1)
    if dropout: conv1 = Dropout(dropoutpct)(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)

    conv2 = Convolution2D(128, (3, 3), padding='same', data_format='channels_last')(pool1)
    conv2 = Activation(activation)(conv2)
    if dropout: conv2 = Dropout(dropoutpct)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Convolution2D(128, (3, 3), padding='same', data_format='channels_last')(conv2)
    conv2 = Activation(activation)(conv2)
    if dropout: conv2 = Dropout(dropoutpct)(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)

    conv3 = Convolution2D(256, (3, 3), padding='same', data_format='channels_last')(pool2)
    conv3 = Activation(activation)(conv3)
    if dropout: conv3 = Dropout(dropoutpct)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Convolution2D(256, (3, 3), padding='same', data_format='channels_last')(conv3)
    conv3 = Activation(activation)(conv3)
    if dropout: conv3 = Dropout(dropoutpct)(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)

    conv4 = Convolution2D(512, (3, 3), padding='same', data_format='channels_last')(pool3)
    conv4 = Activation(activation)(conv4)
    if dropout: conv4 = Dropout(dropoutpct)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Convolution2D(512, (3, 3), padding='same', data_format='channels_last')(conv4)
    conv4 = Activation(activation)(conv4)
    if dropout: conv4 = Dropout(dropoutpct)(conv4)
    conv4 = BatchNormalization()(conv4)

    up1 = UpSampling2D(size=(2, 2))(conv4)
    up1 = concatenate([conv3, up1], axis=3)
    conv5 = Convolution2D(256, (3, 3), padding='same', data_format='channels_last')(up1)
    conv5 = Activation(activation)(conv5)
    if dropout: conv5 = Dropout(dropoutpct)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Convolution2D(256, (3, 3), padding='same', data_format='channels_last')(conv5)
    conv5 = Activation(activation)(conv5)
    if dropout: conv5 = Dropout(dropoutpct)(conv5)
    conv5 = BatchNormalization()(conv5)

    up2 = UpSampling2D(size=(2, 2))(conv5)
    up2 = concatenate([conv2, up2], axis=3)
    conv6 = Convolution2D(128, (3, 3), padding='same', data_format='channels_last')(up2)
    conv6 = Activation(activation)(conv6)
    if dropout: conv6 = Dropout(dropoutpct)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Convolution2D(128, (3, 3), padding='same', data_format='channels_last')(conv6)
    conv6 = Activation(activation)(conv6)
    if dropout: conv6 = Dropout(dropoutpct)(conv6)
    conv6 = BatchNormalization()(conv6)

    up3 = UpSampling2D(size=(2, 2))(conv6)
    up3 = concatenate([conv1, up3], axis=3)
    conv7 = Convolution2D(64, (3, 3), padding='same', data_format='channels_last')(up3)
    conv7 = Activation(activation)(conv7)
    if dropout: conv7 = Dropout(dropoutpct)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Convolution2D(64, (3, 3), padding='same', data_format='channels_last')(conv7)
    conv7 = Activation(activation)(conv7)
    if dropout: conv7 = Dropout(dropoutpct)(conv7)
    conv7 = BatchNormalization()(conv7)

    conv7 = Convolution2D(4, (1, 1))(conv7)
    conv7 = Reshape((img_rows*img_cols, 4))(conv7)
    conv7 = Activation('softmax')(conv7)
    conv7 = Reshape((img_rows, img_cols, 4))(conv7)

    unet_2D = Model(inputs=inputs, outputs=conv7)

    unet_2D.compile(optimizer=sgd, loss=tversky_loss, metrics=[softdice_coef_multilabel])

    return unet_2D


