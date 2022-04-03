from collections import namedtuple, defaultdict
import os
from pathlib import Path

import glob

from PIL import Image
import numpy as np

source_map = {
    'mix': 0,
    'bn': 1,
    'vc': 2,
    'cl': 3,
    'db': 4,
    'fl': 5,
    'hn': 6,
    'ob': 7,
    'sax': 8,
    'tbn': 9,
    'tpt': 10,
    'tba': 11,
    'va': 12,
    'vn': 13}

URMPIndex = namedtuple('URMPIndex', ['datafile', 'offset'])


def _get_source_id_from_filename(filename: str):
    stem = Path(filename).stem
    if stem.startswith('AuSep'):
        instrument_key = stem.split('_')[2]
        return source_map[instrument_key]
    elif stem.startswith('AuMix'):
        return 0
    else:
        print(filename)
        raise IndexError


class URMP:

    def __init__(self):
        self.segment_len = 256
        self.orig_freq_n = 512
        self.log_sample_n = 256
        self.channels_n = 2  # amplitude and phase
        self.sr = 11025
        self.n_instruments = 13
        self.load_specs = True

    def __len__(self):
        return len(self.data_idx)

    def _load_stft_meta_files(self, dataset_dir, phase=False):
        key = '*/*.amp.npy' if not phase else '*/*.phase.npy'
        stft_filenames = glob.glob(os.path.join(dataset_dir, key))

        stft_meta = defaultdict(list)
        for filename in stft_filenames:
            piece_name = Path(filename).parent.stem
            stft_meta[piece_name].append(filename)
        return stft_meta

    def _load_data_from_meta(self, meta: dict):
        data = defaultdict(dict)
        # FixMe reduced memory!!!!
        for piece_name, filenames in list(meta.items()):
            for filename in filenames:
                source_idx = _get_source_id_from_filename(filename)
                data[piece_name][str(source_idx)] = np.load(filename)
            for key in set(str(value) for value in source_map.values()) - data[piece_name].keys():
                data[piece_name][key] = np.zeros(data[piece_name]['0'].shape)
        return data

    def _make_data_idx(self):
        index = list()
        # form an index as list of pairs (piece_name, segment_idx)
        for piece_name, piece_data in self.amp_data.items():
            for i in range(piece_data['0'].shape[1] // self.segment_len):
                index.append((piece_name, i))
        return index

    def save_predicted(self, output_dir, wav_predicted, item_idx, visual_probs=None):
        import librosa
        import pathlib

        train_source_map = ['bn', 'vc', 'cl', 'db', 'fl', 'hn', 'ob', 'sax', 'tbn', 'tpt', 'tba', 'va', 'vn']

        filename, segment_id = self.data_idx[item_idx]
        if not os.path.exists(pathlib.Path(output_dir) / filename):
            os.mkdir(pathlib.Path(output_dir) / filename)

        if visual_probs is not None:
            np.save(pathlib.Path(output_dir) / filename / f'visual_{segment_id:02d}.npy', visual_probs)

        for source_idx, predicted in enumerate(wav_predicted):
            source_name = train_source_map[source_idx]
            predicted_path = pathlib.Path(output_dir) / filename / f'{source_name}_{segment_id:02d}.wav'
            librosa.output.write_wav(predicted_path, predicted, self.sr)


class URMPSpec(URMP):

    def __init__(self, dataset_dir, context=False):
        super(URMPSpec, self).__init__()
        amp_meta = self._load_stft_meta_files(dataset_dir)
        phase_meta = self._load_stft_meta_files(dataset_dir, phase=True)
        self.amp_data = self._load_data_from_meta(amp_meta)
        self.phase_data = self._load_data_from_meta(phase_meta)
        self.data_idx = self._make_data_idx()
        self.multimodal = False
        self.context = context

    def __getitem__(self, item):
        # get piece_name from the index
        piece_name, segment_idx = self.data_idx[item]
        # prepare indexes
        aux_output = np.zeros(self.n_instruments)

        # get segments of the spectrograms according to the index
        data_slice = np.zeros((len(source_map), self.channels_n, self.orig_freq_n, self.segment_len))
        segment_boundaries = slice(segment_idx * self.segment_len, (segment_idx + 1) * self.segment_len)
        for key in self.amp_data[piece_name].keys():
            aux_output[int(key) - 1] = 1
            data_slice[int(key), 0, :, :] = self.amp_data[piece_name][key][:, segment_boundaries]
            data_slice[int(key), 1, :, :] = self.phase_data[piece_name][key][:, segment_boundaries]

        model_input = [data_slice, aux_output] if self.context else data_slice
        return model_input, aux_output


out = URMPSpec[0]

print(out[0], out[0].shape)
print(out[1], out[1].shape)
