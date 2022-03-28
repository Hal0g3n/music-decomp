from __future__ import unicode_literals
import json as _json
import os as _os
import sys as _sys
from functools import partial

import fire as _fire
import youtube_dl as _youtube_dl
import threading
import concurrent.futures
from youtube_dl.utils import ExtractorError, DownloadError

from Solos import SOLOS_IDS_PATH

__all__ = ['YouTubeSaverParallel']


class YouTubeSaverParallel(object):

    def __init__(self):
        self.outtmpl = '%(id)s.%(ext)s'
        self.ydl_opts = {
            'format': 'worstvideo+bestaudio',
            'outtmpl': self.outtmpl,
            """
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            """
            'logger': None
        }

    def download_instrument(self, instrument, dataset_dir="", dataset=""):
        print(instrument)
        if not _os.path.exists(_os.path.join(dataset_dir, instrument)):
            _os.makedirs(_os.path.join(dataset_dir, instrument))
        self.ydl_opts['outtmpl'] = _os.path.join(dataset_dir, instrument, self.outtmpl)
        with _youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
            for i, video_id in enumerate(dataset[instrument]):
                print(fr"Downloading {instrument}/{video_id}")
                try:
                    ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
                except ExtractorError:
                    print('Video unavailable')
                except DownloadError:
                    print('Video unavailable')
                except OSError:
                    with open(_os.path.join(dataset_dir, 'backup.json'), 'w') as dst_file:
                        _json.dump(dataset, dst_file)
                    print(f'Process failed at video {video_id}, #{i}')
                    print(f'Backup saved at {_os.path.join(dataset_dir, "backup.json")}')
                    ydl.download([f'https://www.youtube.com/watch?v={video_id}'])

                except KeyboardInterrupt:
                    _sys.exit()

    def from_json(self, dataset_dir, json_path=SOLOS_IDS_PATH):
        dataset = _json.load(open(json_path))

        with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
            executor.map(partial(self.download_instrument, dataset_dir=dataset_dir, dataset=dataset),
                         ['DoubleBass', 'Horn', 'Oboe', 'Saxophone', 'Trombone', 'Trumpet', 'Tuba', 'Viola', 'Violin'])
            # The other instruments are fully downloaded


if __name__ == '__main__':
    _fire.Fire(YouTubeSaverParallel)

    # USAGE
    # python youtubesaver.py from_json /path_to_your_dst
ys = YouTubeSaverParallel()
ys.from_json(r"C:\Users\User\Documents\GitHub\music-decomp\Solos-Files\data_files\videos")
