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
    def __init__(self, data_dir, mix_no_min=2, training=True, mix_sources_max_no=4, mix_no_max=7, train_test_split=0.8,
                 batch_size=32):

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
        self.dummy_spectrogram_size = (14, 2, 512, 256)  # For tests

        self.metadata = self.load_meta()
        self.window = tf.signal.hann_window(self.ft_hop_size)

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
        return 8000 if self.type == "train" else 2000

    def __generate_data_individual(self):  # This, if it works properly, should basically randomly mix a bunch of sources
        instrument_no = random.randint(self.mix_no_min, self.mix_no_max)
        sources = np.zeros((self.n_instruments + 1, self.audio_len))

        source_indices = np.random.choice(list(range(self.n_instruments)), size=instrument_no, replace=False,
                                          p=np.array(self.source_weights) / sum(self.source_weights))
        # Returns a randomly selected set of indices of sources, with weights given by source weights

        audio_output = np.zeros(self.n_instruments)
        audio_source_indices = [0]

        for instrument in source_indices:  # Note that instrument is an int referrering to the instrument index
            audio_output[instrument] = 1  # One-hot encoded record of which sources are included
            audio_source_indices.append(instrument)  # The paper does instrument +1 here, no idea why
            source_in_question = self.sources[instrument]  # Actual instrument
            sample_selected = random.randrange(
                len(self.data[source_in_question]))  # Picks a random source for the instrument
            start_pos = random.randrange(len(self.data[source_in_question][instrument]) - self.audio_len)

            sources[instrument] = self.data[source_in_question][sample_selected][start_pos:start_pos + self.audio_len]

            smax, smin = sources[instrument].max(), sources[instrument].min()
            # Finds max and min values of that sample
            # This is basically half-assed normalisation
            if not np.isclose((smax - smin), [0.0]):  # If smax not equals smin, basically
                sources[instrument] = (sources[instrument] - smin) / (smax - smin) * 2 - 1
            # TODO Fix this damn normalisation

        sources[self.n_instruments + 1] = sum(sources) / instrument_no
        # More crappy normalisation, you love to see it
        # It's literally the "average loudness", I don't know what to expect

        return sources, audio_output, audio_source_indices

    def __generate_data_batch(self):
        x = np.empty()
