import os

import tensorflow as tf
import keras
import os
from keras import Model
from keras.models import load_model
from pytorch2keras import pytorch_to_keras

from music_decomp_data_generator import SolosDataGenerator
from keras.callbacks import CSVLogger
from unet_implementation.decomp_implimentation import Unet
from music_decomp_data_generator import SolosDataGenerator

# model = Unet().get_model()
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


model = load_model(os.path.abspath(r"..\..\AI\Music Decomposition\saved_models\models_try_2"))
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4 * 2),
#               loss=tf.keras.losses.MeanSquaredError(),
#               metrics=['accuracy', 'mse'])


data_dir_wav = os.path.abspath(r"..\..\Solos-Files\data_files\audio_wav")
print(model.evaluate(SolosDataGenerator(data_dir_wav, training=False)))
