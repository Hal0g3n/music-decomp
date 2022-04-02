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
from librosa import *


class SolosDataGenerator(Sequence):
    def __init__(self, data_dir, mix_no_min=2, training=True, mix_sources_max_no=4, mix_no_max=5, train_test_split=0.8,
                 batch_size=64):
        # The paper sets mix_no_max to 7, but who has 7 different instruments in a normal song

        self.data_dir = data_dir
        self.type = training
        self.multimodal = False
        self.mix_no_min = mix_no_min
        self.mix_no_max = mix_no_max
        self.mix_sources_max_no = mix_sources_max_no
        self.train_test_split = train_test_split
        self.batch_size = batch_size

        self.n_instruments = 13
        self.sources = ['Bassoon', 'Cello', 'Clarinet', 'DoubleBass', 'Flute',
                        'Horn', 'Oboe', 'Saxophone', 'Trombone', 'Trumpet', 'Tuba', 'Viola', 'Violin']

        self.source_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.down_freq = 8000  # Downsample to this frequency
        self.audio_len = 48000  # No of audio samples in each 'snapshot'
        self.ft_window_size = 1022
        self.ft_hop_size = 256
        self.epsilon = 1e-9
        self.log_sample_n = 256  # TODO No idea what this does, I'll figure it out later
        self.segment_len = 256
        self.energy_predicted_sum = 1e-4
        self.dummy_spectrogram_size = (14, 256, 512, 2)
        # Note that the raw spectrogram is of shape (2, 512, 256) and needs to have axes 1 and 3 swapped

        self.metadata = self.load_meta()
        self.window = "hann"

        self.data = {}
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
        # Makes "meta" a dict containing a tuple. The first element is the type of source, i.e. viola, trumpet, etc
        # The second is a sorted list of all files in the data directory that match the pattern of source.wav
        # (.glob is an operation that yields all file paths matching the pattern)

        for source in meta:
            source_len = len(meta[source])
            if self.type:  # True means that it's training
                meta[source] = meta[source][: int(self.train_test_split * source_len)]
            else:
                meta[source] = meta[source][int(self.train_test_split * source_len):]
            # Literally just a slightly clumsy train test split

            for path_index, path in enumerate(meta[source]):
                meta[source][path_index] = (path, sf.info(path).frames)  # .as_posix() doesn't work on WindowsPath
        print(meta)  # For debugging

        return meta

    def load_data(self):
        for source in self.metadata:
            temp = []
            for filename, length in self.metadata[source]:
                temp.append(tf.constant(
                    sf.read(filename)[0]  # Audio stored as a tensor, no idea if this is going to work
                ))
                print(filename)  # For debugging
            self.data[source] = temp.copy()
        print(self.data)

    def __len__(self):
        # No of batches per epoch
        # return 8000 if self.type == "train" else 2000
        return 4  # 256 iterations per epoch is probably enough, right

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
            audio_source_indices.append(instrument)  # The paper does instrument +1 here, no idea why
            source_in_question = self.sources[instrument]  # Actual instrument
            sample_selected = random.randrange(0, len(
                self.data[source_in_question]))  # Picks a random source for the instrument

            start_pos = random.randrange(0, len(self.data[source_in_question][sample_selected]) - self.audio_len)
            sources[instrument] = self.data[source_in_question][sample_selected][start_pos:start_pos + self.audio_len]

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
                sample_stft = librosa.stft(sample[audio_index], n_fft=self.ft_window_size, hop_length=self.ft_hop_size,
                                           window=self.window)
                magnitude, phase = librosa.magphase(sample_stft)
                magnitude = magnitude.T
                phase = phase.T
                spectrograms[source_index, audio_index, :, :, 0] = magnitude + self.energy_predicted_sum
                spectrograms[source_index, audio_index, :, :, 1] = phase
        return spectrograms  # First 13 spectrograms in axis 2 (ok axis 1 but we don't care) are y, the 14th is x

    def __getitem__(self, item):
        # x = np.empty((self.batch_size, 1, 256, 512, 2))
        # y = np.empty((self.batch_size, 13, 256, 512, 2))

        sources = []
        sources_indices = []
        for i in range(self.batch_size):
            source, _, source_index = self.__generate_individual_data()
            sources.append(source)
            sources_indices.append(source_index)

        spectrograms = self.__process_data(sources, sources_indices)

        # x = spectrograms[:, 13:, :, :, :]
        # y = spectrograms[:, :13, :, :, :]

        return spectrograms[:, 13:, :, :, :], spectrograms[:, :13, :, :, :]