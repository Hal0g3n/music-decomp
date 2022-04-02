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

model.summary()

class CustomCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def set_model(self, model):
        self.model = model

saver = CustomCheckpoint(
    filepath=r"\saved_model\model{epoch}",
    save_weights_only=False,
    monitor='val_mse',
    save_best_only=False
)

csv_logger = CSVLogger("\saved_model\model_history_log.csv", append=True)

history = model.fit(x=training_gen,
                    validation_data=validation_gen,
                    steps_per_epoch=len(training_gen),
                    validation_steps=len(validation_gen),
                    use_multiprocessing=False,
                    workers=4,
                    epochs=1000,
                    verbose=1,
                    callbacks=[saver, csv_logger])

