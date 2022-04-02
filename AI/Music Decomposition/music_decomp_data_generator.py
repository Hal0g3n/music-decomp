import re

import librosa
import tensorflow as tf
from keras.utils.data_utils import Sequence
from tensorflow import keras
import json
from pathlib import Path
import random
import numpy as np
import os
import soundfile as sf

class SolosDataGenerator(Sequence):
    def __init__(self, data_dir, mix_no_min=2, training=True, mix_sources_max_no=4, mix_no_max=5, train_test_split=0.8,
                 batch_size=1, load_into_ram=False):
        # The paper sets mix_no_max to 7, but who has 7 different instruments in a normal song

        self.data_dir = data_dir
        self.type = training
        self.multimodal = False
        self.mix_no_min = mix_no_min
        self.mix_no_max = mix_no_max
        self.mix_sources_max_no = mix_sources_max_no
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.load_into_ram = load_into_ram

        self.n_instruments = 13
        self.sources = ['Bassoon', 'Cello', 'Clarinet', 'DoubleBass', 'Flute',
                        'Horn', 'Oboe', 'Saxophone', 'Trombone', 'Trumpet', 'Tuba', 'Viola', 'Violin']

        self.source_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.down_freq = 8000  # Downsample to this frequency
        self.audio_len = 48000  # No of audio samples in each 'snapshot'
        self.ft_window_size = 1022
        self.ft_hop_size = 188  # The paper says that I should use 256, but that just doesn't work
        self.epsilon = 1e-9
        self.log_sample_n = 256  # TODO No idea what this does, I'll figure it out later
        self.energy_predicted_sum = 1e-4
        self.dummy_spectrogram_size = (14, 256, 512, 2)
        # Note that the raw spectrogram is of shape (512, 256) and needs to be transposed

        self.metadata = self.load_meta()
        self.window = "hann"

        self.data = {}
        if self.load_into_ram:
            self.load_data()

        # We will be taking the results seen in papers at face value,
        # and using normalised linearised Fourier Transform Spectrogram, and Wiener post processing,
        # alongside ratio masks and L2 loss

    def increase_n_mix(self):
        if self.mix_no_max < self.mix_sources_max_no:
            self.mix_no_max += 1
            return True
        return False

    def load_meta(self):
        suffix = "*.wav"  # Audio file type, doesn't work on mp3
        # Note that it's mp3 only
        meta = dict(
            [(source, sorted(list((Path(self.data_dir) / (source + "_wav")).glob(suffix)))) for source in self.sources])
        # A little hard to parse, but here we go
        # Makes "meta" a dict. The first key is the type of source, i.e. viola, trumpet, etc
        # The value is a sorted list of all files in the data directory that match the pattern of source.wav
        # (.glob is an operation that yields all file paths matching the pattern)


        for source in meta:
            source_len = len(meta[source])
            if self.type:  # True means that it's training
                meta[source] = meta[source][: int(self.train_test_split * source_len)]
            else:
                meta[source] = meta[source][int(self.train_test_split * source_len):]
            # Literally just a train test split

            for path_index, path in enumerate(meta[source]):
                meta[source][path_index] = (path, sf.info(path).frames)  # .as_posix() doesn't work on WindowsPath

        return meta

    def load_data(self):
        i = 0
        for source in self.metadata:
            i = i + 1
            temp = []
            for filename, length in self.metadata[source]:
                temp.append(tf.constant(
                    sf.read(filename)[0]  # Audio stored as a tensor, no idea if this is going to work
                ))  # For debugging
            self.data[source] = temp.copy()

    def __len__(self):
        # No of batches per epoch
        # return 8000 if self.type == "train" else 2000
        return 16 if self.type else 4  # 512 iterations per epoch is probably enough, right

    def __generate_individual_data(self):
        # This, if it works properly, should basically randomly mix a bunch of sources

        instrument_no = random.randint(self.mix_no_min, self.mix_no_max)
        sources = np.zeros((self.n_instruments + 1, self.audio_len))

        source_indices = np.random.choice(list(range(self.n_instruments)), size=instrument_no, replace=False,
                                          p=np.array(self.source_weights) / sum(self.source_weights))
        # Returns a randomly selected set of indices of sources, with weights given by source weights

        audio_output = np.zeros(self.n_instruments)  # This does not include the extra term below
        audio_source_indices = [
            self.n_instruments]  # This, however, always includes an extra term, that being the average of all inputs

        for instrument in source_indices:  # Note that instrument is an int referring to the instrument index
            audio_output[instrument] = 1  # One-hot encoded record of which sources are included
            audio_source_indices.append(instrument)
            actual_instrument = self.sources[instrument]  # Like, the instrument name
            if not self.load_into_ram:
                filename, length = random.choice(self.metadata[actual_instrument])
                sample_selected = tf.constant(sf.read(filename)[0])
            else:
                sample_index = random.randrange(0, len(
                    self.data[actual_instrument]))  # Picks a random source for the instrument
                sample_selected = self.data[actual_instrument][sample_index]

            start_pos = random.randrange(0, len(sample_selected) - self.audio_len)
            sources[instrument] = sample_selected[start_pos:start_pos + self.audio_len]

            smax, smin = sources[instrument].max(), sources[instrument].min()
            # Finds max and min values of that sample
            # This is basically half-assed normalisation
            if not np.isclose((smax - smin), [0.0]):  # If smax not equals smin, basically
                sources[instrument] = (sources[instrument] - smin) / (smax - smin) * 2 - 1
            # TODO Fix this damn normalisation

        sources[self.n_instruments] = sum(sources) / instrument_no
        # Smushed waveforms, you can just do that with wav, it turns out

        # Note that audio_output is 0 indexed

        return sources, audio_output, audio_source_indices

    def __process_data(self, sources, sources_indices):
        # Where sources is an list of batch_size samples
        # And Source_indices is a list of arrays of what instruments are in the source
        spectrograms = np.zeros((self.batch_size, *self.dummy_spectrogram_size))

        for source_index, sample in enumerate(sources):
            for audio_index in sources_indices[source_index]:
                self.ft_hop_size = 188
                sample_stft = librosa.stft(sample[audio_index], n_fft=self.ft_window_size, hop_length=self.ft_hop_size,
                                           window=self.window)
                magnitude, phase = librosa.magphase(sample_stft)
                magnitude = magnitude.T
                phase = phase.T
                spectrograms[source_index, audio_index, :, :, 0] = magnitude + self.energy_predicted_sum
                spectrograms[source_index, audio_index, :, :, 1] = phase
        self.spectrograms = spectrograms
        # First 13 spectrograms in axis 2 (ok axis 1 but we don't care) are y, the 14th is x

    def _compute_masks(self):

        sources = self.spectrograms[:, :13, :, :, 0]
        x = self.spectrograms[:, 13, :, :, :1]
        # x = np.expand_dims(x, axis=1)

        y = sources / np.expand_dims(np.sum(sources, axis=1), axis=1)
        y = np.swapaxes(y, 1, 2)
        y = np.swapaxes(y, 2, 3)
        return x, y

    def __getitem__(self, item):
        # x = np.empty((self.batch_size, 1, 256, 512, 2))
        # y = np.empty((self.batch_size, 13, 256, 512, 2))

        sources = []
        sources_indices = []
        for i in range(self.batch_size):
            source, _, source_index = self.__generate_individual_data()
            sources.append(source)
            sources_indices.append(source_index)

        self.__process_data(sources, sources_indices)

        # x = spectrograms[:, 13:, :, :, :]
        # y = spectrograms[:, :13, :, :, :]

        return self._compute_masks()
