from keras import layers, Model, backend
import numpy as np
from functools import partial


class Unet(Model):
    def get_config(self):
        super(self).get_config()

    def __init__(self):
        super().__init__()
        self.inputs = layers.Input(shape=(512, 512, 1))
        self.layer_1 = partial(self.downsample, n_filters=16, dropout=True)
        self.layer_2 = partial(self.downsample, n_filters=32, dropout=True)
        self.layer_3 = partial(self.downsample, n_filters=64, dropout=True)
        self.layer_4 = partial(self.downsample, n_filters=128, dropout=False)
        self.layer_5 = partial(self.downsample, n_filters=256, dropout=False)
        self.layer_6 = partial(self.downsample, n_filters=512, dropout=False)

        self.layer_7_1 = layers.LeakyReLU()
        self.layer_7_2 = layers.Conv2D(512, (4, 4), padding="same", kernel_initializer="he_normal")

        self.layer_8 = partial(self.upsample, n_filters=512, dropout=False)
        self.layer_9 = partial(self.upsample, n_filters=256, dropout=False)
        self.layer_10 = partial(self.upsample, n_filters=128, dropout=False)
        self.layer_11 = partial(self.upsample, n_filters=64, dropout=True)
        self.layer_12 = partial(self.upsample, n_filters=32, dropout=True)
        self.layer_13 = partial(self.upsample, n_filters=16, dropout=True)

    def call(self, x, training=None, mask=None):
        if training is None:
            training = backend.learning_phase()
        x, out_1 = self.layer_1(x, training=training)
        print(x.shape)
        x, out_2 = self.layer_2(x, training=training)
        print(x.shape)
        x, out_3 = self.layer_3(x, training=training)
        print(x.shape)
        x, out_4 = self.layer_4(x, training=training)
        print(x.shape)
        x, out_5 = self.layer_5(x, training=training)
        print(x.shape)
        x, out_6 = self.layer_6(x, training=training)
        print(x.shape)

        x = self.layer_7_1(x)
        x = self.layer_7_2(x)
        print(x.shape)
        print(out_6.shape)

        x = self.layer_8(x, out_6, training=training)
        print(x.shape)
        x = self.layer_9(x, out_5, training=training)
        print(x.shape)
        x = self.layer_10(x, out_4, training=training)
        print(x.shape)
        x = self.layer_11(x, out_3, training=training)
        print(x.shape)
        x = self.layer_12(x, out_2, training=training)
        print(x.shape)
        x = self.layer_13(x, out_1, training=training)
        print(x.shape)
        output = layers.LeakyReLU()(x)
        output = layers.BatchNormalization()(output, training)
        output = layers.Conv2D(16, (3, 3), padding="same", kernel_initializer="he_normal")(output)
        output = layers.BatchNormalization()(output, training)
        output = layers.Conv2D(13, (1, 1))(output)
        # This last block is just the end, finishes up translating to an output
        # I have no idea if this does anything, but it seems like it should
        print(output.shape)
        return output

    def downsample(self, x, n_filters=16, training=False, dropout=False):
        orig_features = layers.LeakyReLU()(x)
        pass_on = layers.ZeroPadding2D(padding=1)(orig_features)
        pass_on = layers.Conv2D(n_filters, (4, 4), strides=2, padding="valid",
                                kernel_initializer="he_normal")(pass_on)

        pass_on = layers.BatchNormalization()(pass_on, training)
        if dropout:
            pass_on = layers.Dropout(0.4)(pass_on, training)
        return pass_on, orig_features

    def upsample(self, x, orig_features, n_filters=16, training=False, dropout=False):
        upsampled = layers.LeakyReLU()(x)
        upsampled = layers.UpSampling2D(size=2, interpolation="bilinear")(upsampled)
        upsampled = layers.Conv2DTranspose(n_filters, (3, 3), strides=1, padding="same",
                                           kernel_initializer="he_normal")(upsampled)
        upsampled = layers.BatchNormalization()(upsampled, training)
        upsampled = layers.concatenate((upsampled, orig_features), axis=3)
        if dropout:
            upsampled = layers.Dropout(0.4)(upsampled, training)
        return upsampled


model = Unet()
print(model.__call__(np.random.random((1, 256, 512, 1)), training=False))  # Yes I know I could just call it, but this makes it obvious what's happening
model.summary()
