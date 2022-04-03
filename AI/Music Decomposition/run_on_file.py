import os

import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import soundfile as sf
from keras.models import load_model
from matplotlib import pyplot as plt

from unet_implementation.decomp_implimentation import Unet
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError


class ModelPredictionWrapper:
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Necessary in case he has a GPU set up, this turbo breaks
        self.ft_hop_size = 188
        self.window = "hann"
        self.ft_window_size = 1022
        self.dummy_spectrogram_size = (14, 256, 512, 2)
        ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..' + os.sep + '..'))
        self.temp_dir = fr"{ROOT}\UI\temp"
        self.model = load_model(fr"{ROOT}\AI\Music Decomposition\saved_models\models_try_2")
        os.makedirs(self.temp_dir, exist_ok=True)

    def file_to_wav(self, filename):
        input_format = filename.split(".")[-1]
        try:
            audio = AudioSegment.from_file(filename, format=input_format)
            audio.set_frame_rate(8000)  # My model probably breaks if I don't do this
            audio.export(self.temp_dir + fr"\master_file.wav", format="wav")

        except CouldntDecodeError as e:
            print("INVALID FILE FORMAT")
            # TODO Yuan xi handle this error

    def get_chunks(self):
        audio_file = np.array(sf.read(self.temp_dir + fr"\master_file.wav")[0])
        # Yep, I'm just loading the whole damn thing into memory, stop me
        length = audio_file.shape[0]
        chunks = length // 48000
        of_chunks = []
        for i in range(chunks):
            of_chunks.append(audio_file[i * 48000:(i + 1) * 48000])
        if length % 48000 == 0:
            padding_needed = 0
        else:
            padding_needed = 48000 - length % 48000
        of_chunks.append(np.pad(audio_file[chunks * 48000:], (0, padding_needed)))

        return of_chunks

    def chunks_to_spectograms(self, chunks):
        spectrograms = np.zeros((len(chunks), *(256, 512, 2)))
        complex_spectrograms = np.zeros((len(chunks), *(256, 512, 1)), dtype = 'complex_')

        for index, chunk in enumerate(chunks):
            sample_stft = librosa.stft(chunk, n_fft=self.ft_window_size, hop_length=self.ft_hop_size,
                                       window=self.window)
            magnitude, phase = librosa.magphase(sample_stft)
            magnitude = magnitude.T
            phase = phase.T
            spectrograms[index, :, :, 0] = magnitude + 1e-4
            spectrograms[index, :, :, 1] = phase
            complex_spectrograms[index, :, :, 0] = sample_stft.T
        return complex_spectrograms, spectrograms, len(chunks)

    def call_model(self):
        complex_spectrograms, spectrograms, chunk_count = self.chunks_to_spectograms(self.get_chunks())
        out_spectrograms = np.zeros((chunk_count, 256, 512, 13))
        preds = self.model.predict(spectrograms[:, :, :, 0], batch_size=chunk_count)
        print(preds.shape)
        print(spectrograms.shape)
        print(complex_spectrograms.shape)
        for i in range(chunk_count):
            out_spectrograms[i, :, :, :] = complex_spectrograms[i, :, :, :] * preds[i, :, :, :]
        fig, ax = plt.subplots(1, 2)
        D = librosa.amplitude_to_db(np.abs(out_spectrograms[0, :, :, :]),
                                    ref=np.max)
        librosa.display.specshow(D, y_axis='log', sr=8000, hop_length=self.ft_hop_size, n_fft=self.ft_window_size,
                                 x_axis='time', ax=ax[0])
        fig.show()


mpw = ModelPredictionWrapper()
mpw.file_to_wav(r"C:\Users\User\Documents\GitHub\music-decomp\data\01_Jupiter_vn_vc\AuMix_01_Jupiter_vn_vc.wav")
mpw.call_model()
