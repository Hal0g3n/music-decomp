import os
from os import listdir
from os.path import isfile, join
from pydub import AudioSegment
from path import Path

for i in ['Bassoon', 'Cello', 'Clarinet', 'DoubleBass', 'Flute',
                        'Horn', 'Oboe', 'Saxophone', 'Trombone', 'Trumpet', 'Tuba', 'Viola', 'Violin']:
    ROOT = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..' + os.sep + '..'))
    print(ROOT)
    OR_PATH = fr"{ROOT}\Solos-Files\data_files\audio\{i}_Audio"
    DST_PATH = fr"{ROOT}\Solos-Files\data_files\audio\{i}_wav"
    #files = [f for f in listdir(OR_PATH) if isfile(join(OR_PATH, f))]
    files = [mp3_file for mp3_file in Path(ROOT + fr"\Solos-Files\data_files\audio\{i}_Audio").glob("*.mp3")]
    for file in files:
        print(file)
        #audio = AudioSegment.from_mp3(file)
        print(os.path.abspath(fr"..\..\Solos-Files\data_files\audio\Bassoon_Audio\-0yEIJCnno8.mp3"))
        print(os.path.exists(os.path.abspath(fr"..\..\Solos-Files\data_files\audio\Bassoon_Audio\-0yEIJCnno8.mp3")))
        audio = AudioSegment.from_mp3(fr"C:\Users\Vikram Ramanathan\Documents\GitHub\music-decomp\Solos-Files\data_files\audio\Bassoon_Audio\-0yEIJCnno8.mp3")
        audio.export(f"{file[:4]}.wav", format="wav")
        break
    print(files)

    break

