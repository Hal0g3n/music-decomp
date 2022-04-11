from keras import layers, Model, backend
from functools import partial


class Unet:

    def __init__(self):
        self.inputs = layers.Input(shape=(256, 512, 1))
        self.layer_1 = partial(self.downsample, n_filters=64, dropout=True)
        self.layer_2 = partial(self.downsample, n_filters=128, dropout=True)
        self.layer_3 = partial(self.downsample, n_filters=256, dropout=True)
        self.layer_4 = partial(self.downsample, n_filters=512, dropout=True)
        self.layer_5 = partial(self.downsample, n_filters=512, dropout=True)

        self.layer_6 = partial(self.upsample, n_filters=512, dropout=True)
        self.layer_7 = partial(self.upsample, n_filters=512, dropout=True)
        self.layer_8 = partial(self.upsample, n_filters=256, dropout=True)
        self.layer_9 = partial(self.upsample, n_filters=128, dropout=True)
        self.layer_10 = partial(self.upsample, n_filters=64, dropout=True)

    def get_model(self):
        x = layers.BatchNormalization()(self.inputs)
        x, out_1 = self.layer_1(x)
        x, out_2 = self.layer_2(x)
        x, out_3 = self.layer_3(x)
        x, out_4 = self.layer_4(x)
        x, out_5 = self.layer_5(x)

        x = self.layer_6(x, out_5)
        x = self.layer_7(x, out_4)
        x = self.layer_8(x, out_3)
        x = self.layer_9(x, out_2)
        x = self.layer_10(x, out_1)
        output = layers.ReLU()(x)
        output = layers.Conv2D(16, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer="l2")(output)
        output = layers.Conv2D(14, (1, 1))(output)
        # This last block is just the end, finishes up translating to an output
        # I have no idea if this does anything, but it seems like it should
        model = Model(self.inputs, output)
        return model

    def downsample(self, x, n_filters=16, dropout=False):
        orig_features = layers.LeakyReLU(0.2)(x)
        pass_on = layers.Conv2D(n_filters, (4, 4), strides=2, padding="same",
                                kernel_initializer="he_normal")(orig_features)

        pass_on = layers.BatchNormalization()(pass_on)
        if dropout:
            pass_on = layers.Dropout(0.2)(pass_on)
        return pass_on, orig_features

    def upsample(self, x, orig_features, n_filters=16, dropout=False):
        upsampled = layers.ReLU()(x)
        upsampled = layers.UpSampling2D(size=2, interpolation="bilinear")(upsampled)
        upsampled = layers.Conv2D(n_filters, (3, 3), strides=1, padding="same",
                                           kernel_initializer="he_normal")(upsampled)
        upsampled = layers.BatchNormalization()(upsampled)
        upsampled = layers.concatenate((upsampled, orig_features), axis=3)
        if dropout:
            upsampled = layers.Dropout(0.2)(upsampled)
        return upsampled
