"""Compute input examples for VGGish from audio waveform."""
import os

import numpy as np
import pandas as pd
import resampy
from scipy.io import wavfile

import utils.signal_utils as signal
from params import vggish_params
from utils import data_utils


def waveform_to_examples(data, sample_rate):
    """Converts audio waveform into an array of examples for VGGish.

    Args:
      data: np.array of either one dimension (mono) or two dimensions
        (multi-channel, with the outer dimension representing channels).
        Each sample is generally expected to lie in the range [-1.0, +1.0],
        although this is not required.
      sample_rate: Sample rate of data.

    Returns:
      3-D np.array of shape [num_examples, num_frames, num_bands] which represents
      a sequence of examples, each of which contains a patch of log mel
      spectrogram, covering num_frames frames of audio and num_bands mel frequency
      bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.
    """
    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    # Resample to the rate assumed by VGGish.
    if sample_rate != vggish_params.SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

    # Compute log mel spectrogram features.
    log_mel = signal.log_mel_spectrogram(
        data,
        audio_sample_rate=vggish_params.SAMPLE_RATE,
        log_offset=vggish_params.LOG_OFFSET,
        window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=vggish_params.NUM_MEL_BINS,
        lower_edge_hertz=vggish_params.MEL_MIN_HZ,
        upper_edge_hertz=vggish_params.MEL_MAX_HZ)

    # Frame features into examples.
    features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(
        vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(
        vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples = signal.frame(
        log_mel,
        window_length=example_window_length,
        hop_length=example_hop_length)
    return log_mel_examples


def wavfile_to_examples(wav_file):
    """Convenience wrapper around waveform_to_examples() for a common WAV format.

    Args:
      wav_file: String path to a file, or a file-like object. The file
      is assumed to contain WAV audio data with signed 16-bit PCM samples.

    Returns:
      See waveform_to_examples.
    """
    sr, wav_data = wavfile.read(wav_file)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    return waveform_to_examples(samples, sr)


def read_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("File `%s` not exists." % file_path)

    with open(file_path) as f:
        df = pd.read_csv(f)

    return df


if __name__ == '__main__':
    feature_path = "../data/features"
    label_path = "../data/labels"

    # files_dir = r"E:\机器学习\数据集\音频\whale_data\Audio\train"
    # files = os.listdir(files_dir)
    # files.sort(key=lambda x: int(re.match(r"train(\d+).wav", x).group(1)))
    # features = []
    # for file in tqdm(files):
    #     path = os.path.join(files_dir, file)
    #     feature = wavfile_to_examples(path)
    #     feature = np.reshape(
    #         feature, (vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS))
    #     features.append(feature)
    #
    # data_utils.save_data(np.array(features), feature_path)

    df = read_csv(r'E:\机器学习\数据集\音频\whale_data\labels.csv')

    labels = []
    for label in df.label:
        labels.append(label)

    data_utils.save_data(np.array(labels), label_path)
