import os

from pydub.audio_segment import AudioSegment
from tqdm import tqdm


def convert_audio(from_path,
                  to_dir,
                  from_format=None,
                  to_format='wav'):
    """Convert audios to a specified audio format."""
    assert os.path.isdir(to_dir)

    from_files = []
    if os.path.isfile(from_path):
        from_files.append(from_path)
    elif os.path.isdir(from_path):
        file_paths = [os.path.join(from_path, name) for name in os.listdir(from_path)]
        from_files = [file for file in file_paths if os.path.isfile(file)]

    for from_file in tqdm(from_files):
        to_name = os.path.basename(from_file). \
            replace(os.path.splitext(from_file)[1], '.' + to_format)
        to_path = os.path.join(to_dir, to_name)
        audio = AudioSegment.from_file(from_file, from_format)
        audio.export(to_path, to_format)


if __name__ == '__main__':
    _from_path = r'E:\机器学习\数据集\音频\whale_data\whale_data\data\train'
    _to_dir = r'E:\机器学习\数据集\音频\whale_data\Audio\train'
    convert_audio(_from_path, _to_dir)
