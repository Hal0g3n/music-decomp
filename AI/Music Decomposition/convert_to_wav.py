import os
from os import listdir
from os.path import isfile, join
from pydub import AudioSegment
from path import Path
import multiprocessing as mp


def convert(instrument, ROOT):
    OR_PATH = fr"{ROOT}\Solos-Files\data_files\audio\{instrument}_Audio"
    DST_PATH = fr"{ROOT}\Solos-Files\data_files\audio_wav\{instrument}_wav"

    os.makedirs(DST_PATH, exist_ok=True)
    files = [mp3_file for mp3_file in Path(OR_PATH).glob("*.mp3")]
    for file in files:
        print(file)
        audio = AudioSegment.from_file(file, format="mp3")
        audio.set_frame_rate(8000)  # We're downsampling to 8000 anyway so
        name = file.stem  # This is the filename
        print(name)
        out_file = DST_PATH + fr"\{name}.wav"
        audio.export(out_file, format="wav")


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..' + os.sep + '..'))
if __name__ == '__main__':
    pool = mp.Pool(6)
    results = [pool.apply(convert, args=(instrument, ROOT)) for instrument in
               ['Bassoon', 'Cello', 'Clarinet', 'DoubleBass', 'Flute',
                'Horn', 'Oboe', 'Saxophone', 'Trombone', 'Trumpet', 'Tuba', 'Viola', 'Violin']]
    pool.close()
