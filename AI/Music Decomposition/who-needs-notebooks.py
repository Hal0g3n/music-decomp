import pickle

import tensorflow as tf
from keras.models import load_model
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
from unet_implementation.decomp_implimentation_scuffed import Unet

print("Imported 2")
data_dir_wav = os.path.abspath(r"..\..\Solos-Files\data_files\audio_wav")

training_gen = SolosDataGenerator(data_dir_wav, training=True)
validation_gen = SolosDataGenerator(data_dir_wav, training=False)
print("Generators Initialized")

#model = Unet().get_model()
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4 * 2),
#              loss=tf.keras.losses.MeanSquaredError(),
#              metrics=['mse'])

model = load_model(os.path.abspath(r"..\..\AI\Music Decomposition\saved_models\model_scuffed"))
print("Model Compiled")


# model.summary()


class CustomCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def set_model(self, model):
        self.model = model


saver = CustomCheckpoint(
    filepath=os.path.abspath(
        r"C:\Users\User\Documents\GitHub\music-decomp\AI\Music Decomposition\saved_models\model_scuffed"),
    save_weights_only=True,
    monitor='val_mse',
    save_best_only=False,
    save_freq="epoch"
)
print(r"C:\Users\User\Documents\GitHub\music-decomp\AI\Music Decomposition\saved_models\model_scuffed")
#model.save(r"C:\Users\User\Documents\GitHub\music-decomp\AI\Music Decomposition\saved_models\model_scuffed")
csv_logger = CSVLogger(os.path.abspath(
    r"C:\Users\User\Documents\GitHub\music-decomp\AI\Music Decomposition\saved_models\model_history_log_scuffed.csv"),
                       append=True)


def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    return 5e-5


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
# First model trained with batch = 8, second with batch = 4, third with batch = 16 again, and scuffed with batch = 4
model.save(r"C:\Users\User\Documents\GitHub\music-decomp\AI\Music Decomposition\saved_models\model_scuffed")
model.save_weights(r"C:\Users\User\Documents\GitHub\music-decomp\AI\Music Decomposition\saved_models\model_scuffed")
