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
import os

from music_decomp_data_generator import SolosDataGenerator

print("Imported 1")
from unet_implementation.decomp_implimentation import Unet

print("Imported 2")
data_dir_wav = os.path.abspath(r"..\..\Solos-Files\data_files\audio_wav")

training_gen = SolosDataGenerator(data_dir_wav, training=True)
validation_gen = SolosDataGenerator(data_dir_wav, training=False)

model = Unet()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4 * 2),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy', 'mse'])

model.fit(x=training_gen,
          validation_data=validation_gen,
          validation_steps=len(training_gen),
          use_multiprocessing=True,
          workers=6,
          epochs=1)
