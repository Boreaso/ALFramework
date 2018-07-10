import os
import pickle
import re


def to_csv(files):
    for file in files:
        df = pickle.load(open(file, mode='rb'))
        df.to_csv(file[:-2] + 'csv', index=False)


def list_files(top, pattern):
    result = []
    for root, _, files in os.walk(top):
        for file in files:
            if re.fullmatch(pattern, file) is not None:
                full_path = os.path.join(root, file)
                result.append(full_path)
    return result


def del_files(files):
    for file in files:
        os.remove(file)


if __name__ == '__main__':
    files = list_files('../outputs', '.*.pd')
    to_csv(files)

    del_files(files)
