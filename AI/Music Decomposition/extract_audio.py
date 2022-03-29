from flerken.video.utils import apply_tree, apply_single

import os
import subprocess

def extract_audio(OR_PATH, DST_PATH):

    rep = apply_tree(OR_PATH, DST_PATH,
                     output_options=['-ac', '1', '-ar', '16000'],
                     multiprocessing=0,
                     ext='.wav',
                     fn=apply_single)


for i in ['Bassoon', 'Cello', 'Clarinet', 'DoubleBass', 'Flute',
                        'Horn', 'Oboe', 'Saxophone', 'Trombone', 'Trumpet', 'Tuba', 'Viola', 'Violin']:
    OR_PATH = fr"C:\Users\User\Documents\GitHub\music-decomp\Solos-Files\data_files\audio\{i}_Audio"
    DST_PATH = fr"C:\Users\User\Documents\GitHub\music-decomp\Solos-Files\data_files\audio\{i}_wav"
    print(i)
    extract_audio(
        OR_PATH,
        DST_PATH)

