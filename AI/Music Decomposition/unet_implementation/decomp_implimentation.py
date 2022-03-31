import tensorflow
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from keras import activations


class Unet(Model):
    def __init__(self, training=True):
        super().__init__()
        self.training = training
        self.inputs = layers.Input(shape=(512, 512, 1))
        self.layer_1 = partial(self.downsample, n_filters=16, dropout=True)
        self.layer_2 = partial(self.downsample, n_filters=32, dropout=True)
        self.layer_3 = partial(self.downsample, n_filters=64, dropout=True)
        self.layer_4 = partial(self.downsample, n_filters=128, dropout=False)
        self.layer_5 = partial(self.downsample, n_filters=256, dropout=False)
        self.layer_6 = partial(self.downsample, n_filters=512, dropout=False)

        self.layer_7 = layers.Conv2D(1024, (4, 4), output_padding=1, activation="relu",
                                     kernel_initializer="he_normal")

        self.layer_8 = partial(self.upsample, n_filters=512, dropout=False)
        self.layer_9 = partial(self.upsample, n_filters=256, dropout=False)
        self.layer_10 = partial(self.upsample, n_filters=128, dropout=False)
        self.layer_11 = partial(self.upsample, n_filters=64, dropout=True)
        self.layer_12 = partial(self.upsample, n_filters=32, dropout=True)
        self.layer_13 = partial(self.upsample, n_filters=16, dropout=True)

    def call(self, x):
        x = self.layer_1(x, self.training)
        print(x.shape)
        x = self.layer_2(x, self.training)
        print(x.shape)
        x = self.layer_3(x, self.training)
        print(x.shape)
        x = self.layer_4(x, self.training)
        print(x.shape)
        x = self.layer_5(x, self.training)
        print(x.shape)
        x = self.layer_6(x, self.training)
        print(x.shape)

        x = self.layer_7(x, self.training)
        print(x.shape)

        x = self.layer_8(x, self.training)
        print(x.shape)
        x = self.layer_9(x, self.training)
        print(x.shape)
        x = self.layer_10(x, self.training)
        print(x.shape)
        x = self.layer_11(x, self.training)
        print(x.shape)
        x = self.layer_12(x, self.training)
        print(x.shape)
        x = self.layer_13(x, self.training)
        print(x.shape)
        return x


    def downsample(self, x, n_filters, dropout=False):
        pass_on = layers.LeakyReLU()(x)
        pass_on = layers.Conv2D(n_filters, (4, 4), output_padding=1, strides=2, activation="relu",
                                kernel_initializer="he_normal")(pass_on)

        pass_on = layers.BatchNormalization(pass_on, self.training)
        if dropout:
            pass_on = layers.Dropout(0.4)(pass_on)
        return pass_on

    def upsample(self, x, orig_features, n_filters, dropout=False):
        upsampled = layers.LeakyReLU()(x)
        upsampled = layers.Conv2DTranspose(n_filters, (3, 3), output_padding=1, strides=1,
                                           kernel_initializer="he_normal")(upsampled)
        upsampled = layers.BatchNormalization(upsampled, self.training)
        upsampled = layers.UpSampling2D(2)(upsampled)
        upsampled = layers.concatenate((upsampled, orig_features))
        if dropout:
            upsampled = layers.Dropout(0.5)(upsampled)
        return upsampled
