from keras.models import Model
from keras.layers import Input, concatenate, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D
from keras.optimizers import SGD
import keras

def get_unet_2D(img_rows, img_cols):
    inputs = Input(shape=(img_rows, img_cols, 1)) #(1, dim1, dim2, channels)
    #print(inputs.shape)

    conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(inputs)
    conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)

    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(pool1)
    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)

    conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last')(pool2)
    conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)

    conv4 = Convolution2D(256, (3, 3), activation='relu', padding='same', data_format='channels_last')(pool3)
    conv4 = Convolution2D(256, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv4)

    up1 = UpSampling2D(size=(2, 2))(conv4)
    up1 = concatenate([conv3, up1], axis=3)
    conv5 = Convolution2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last')(up1)
    conv5 = Convolution2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv5)

    up2 = UpSampling2D(size=(2, 2))(conv5)
    up2 = concatenate([conv2, up2], axis=3)
    conv6 = Convolution2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(up2)
    conv6 = Convolution2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv6)

    up3 = UpSampling2D(size=(2, 2))(conv6)
    up3 = concatenate([conv1, up3], axis=3)
    conv7 = Convolution2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(up3)
    conv7 = Convolution2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv7)

    conv7 = Convolution2D(4, (1, 1), activation='softmax')(conv7)

    unet_2D = Model(inputs=inputs, outputs=conv7)

    sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    unet_2D.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return unet_2D

def get_unet_3D(img_rows, img_cols, img_slices):
    inputs = Input(shape=(img_rows, img_cols, img_slices, 1)) #(1, dim1, dim2, dim3, channels)
    print(inputs.shape)

    conv1 = Convolution3D(10, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(inputs)
    conv1 = Convolution3D(10, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv1)

    conv2 = Convolution3D(20, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(pool1)
    conv2 = Convolution3D(20, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)

    conv3 = Convolution3D(40, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(pool2)
    conv3 = Convolution3D(40, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv3)

    conv4 = Convolution3D(80, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(pool3)
    conv4 = Convolution3D(80, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv4)

    up1 = UpSampling3D(size=(2, 2, 2))(conv4)
    up1 = concatenate([conv3, up1], axis=4)
    conv5 = Convolution3D(40, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(up1)
    conv5 = Convolution3D(40, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv5)

    up2 = UpSampling3D(size=(2, 2, 2))(conv5)
    up2 = concatenate([conv2, up2], axis=4)
    conv6 = Convolution3D(20, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(up2)
    conv6 = Convolution3D(20, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv6)

    up3 = UpSampling3D(size=(2, 2, 2))(conv6)
    up3 = concatenate([conv1, up3], axis=4)
    conv7 = Convolution3D(10, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(up3)
    conv7 = Convolution3D(10, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv7)

    conv7 = Convolution3D(4, (1, 1, 1), activation='softmax')(conv7)

    unet_3D = Model(inputs=inputs, outputs=conv7)

    sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    unet_3D.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return unet_3D
