import argparse
from collections import Counter
import glob
import hashlib
import json
import os
import librosa

ORCHSET_INDEX_PATH = '../mirdata/indexes/orchset_index.json'

def audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=False)
    num_channels = 1 if y.ndim == 1 else y.shape[0]
    length = y.shape[-1]/sr
    return length, sr, num_channels


def summarize_audio(audio_records):
    return {
        "file_count": len(audio_records),
        "total_audio_duration_seconds": sum(
            record["duration_seconds"] for record in audio_records
        ),
        "sample_rate_counts": dict(
            Counter(record["sample_rate"] for record in audio_records)
        ),
        "channel_counts": dict(
            Counter(record["num_channels"] for record in audio_records)
        ),
    }


def md5(file_path):
    """Get md5 hash of a file.

    Parameters
    ----------
    file_path: str
        File path.

    Returns
    -------
    md5_hash: str
        md5 hash of data in file_path
    """
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def make_orchset_index(data_path):

    mono_audio_files = sorted(
        glob.glob(os.path.join(data_path, 'audio', 'mono', '*.wav'))
    )

    index = {}
    audio_records = []
    for audio_path in mono_audio_files:
        track_id = os.path.basename(audio_path).split('.')[0]

        audio_stereo_checksum = md5(
            os.path.join(data_path, 'audio', 'stereo', '{}.wav'.format(track_id))
        )
        audio_mono_checksum = md5(
            os.path.join(data_path, 'audio', 'mono', '{}.wav'.format(track_id))
        )
        melody_checksum = md5(os.path.join(data_path, 'GT', '{}.mel'.format(track_id)))

        length_mono, sr_mono, num_channels_mono = audio_features(os.path.join(data_path, 'audio', 'mono', '{}.wav'.format(track_id)))
        length_stereo, sr_stereo, num_channels_stereo = audio_features(os.path.join(data_path, 'audio', 'stereo', '{}.wav'.format(track_id)))
        for duration, sample_rate, channels in (
            (length_mono, sr_mono, num_channels_mono),
            (length_stereo, sr_stereo, num_channels_stereo),
        ):
            audio_records.append({
                "duration_seconds": duration,
                "sample_rate": sample_rate,
                "num_channels": channels,
            })
        index[track_id] = {
            'audio_stereo': (
                'audio/stereo/{}.wav'.format(track_id),
                audio_stereo_checksum,
            ),
            'length_stereo': (length_stereo),
            'sr_stereo' : (sr_stereo),
            'num_channels_stereo' : (num_channels_stereo),
            'audio_mono': ('audio/mono/{}.wav'.format(track_id), audio_mono_checksum),
            'melody': ('GT/{}.mel'.format(track_id), melody_checksum),
            'length_mono': (length_mono),
            'sr_mono' : (sr_mono),
            'num_channels_mono': (num_channels_mono)
        }
    audio_summary = summarize_audio(audio_records)

    index["audio_summary"] = audio_summary

    with open(ORCHSET_INDEX_PATH, 'w') as fhandle:
        json.dump(index, fhandle, indent=2)

    print(json.dumps(audio_summary, indent=2))


def main(args):
    make_orchset_index(args.orchset_data_path)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Make Orchset index file.')
    PARSER.add_argument(
        'orchset_data_path', type=str, help='Path to Orchset data folder.'
    )

    main(PARSER.parse_args())
