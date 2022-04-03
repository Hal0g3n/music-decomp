import pickle

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras.optimizers import Adam
import json
from pathlib import Path
import random
import numpy as np
import librosa
import os
import soundfile as sf
from keras.utils.data_utils import Sequence
import matplotlib.pyplot as plt
import os
from music_decomp_data_generator import SolosDataGenerator
from keras.callbacks import CSVLogger

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("Imported 1")
from unet_implementation.decomp_implimentation import Unet

print("Imported 2")
data_dir_wav = os.path.abspath(r"..\..\Solos-Files\data_files\audio_wav")

training_gen = SolosDataGenerator(data_dir_wav, training=True)
validation_gen = SolosDataGenerator(data_dir_wav, training=False)
print("Generators Initialized")

model = Unet().get_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4 * 2),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy', 'mse'])
print("Model Compiled")


# model.summary()


class CustomCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def set_model(self, model):
        self.model = model


saver = CustomCheckpoint(
    filepath=os.path.abspath(
        r"C:\Users\User\Documents\GitHub\music-decomp\AI\Music Decomposition\saved_models\model_try_3"),
    save_weights_only=True,
    monitor='val_mse',
    save_best_only=False,
    save_freq="epoch"
)
print(r"C:\Users\User\Documents\GitHub\music-decomp\AI\Music Decomposition\saved_models\models_try_3")
model.save(r"C:\Users\User\Documents\GitHub\music-decomp\AI\Music Decomposition\saved_models\models_try_3")
csv_logger = CSVLogger(os.path.abspath(
    r"C:\Users\User\Documents\GitHub\music-decomp\AI\Music Decomposition\saved_models\model_history_log.csv"),
                       append=True)


def scheduler(epoch, lr):
    if epoch < 20:
        return lr
    return lr * 0.98


print(model)
# TODO stop hardcoding file paths

history = model.fit(x=training_gen,
                    validation_data=validation_gen,
                    steps_per_epoch=len(training_gen),
                    validation_steps=len(validation_gen),
                    use_multiprocessing=False,
                    workers=8,
                    epochs=100,
                    verbose=1,
                    callbacks=[saver, csv_logger, tf.keras.callbacks.LearningRateScheduler(scheduler)])
# First model trained with batch = 8, second with batch = 4, third with batch = 16 again
model.save(r"C:\Users\User\Documents\GitHub\music-decomp\AI\Music Decomposition\saved_models\models_try_3")
model.save_weights(r"C:\Users\User\Documents\GitHub\music-decomp\AI\Music Decomposition\saved_models\models_try_3")
