import csv
import os
import pickle

import numpy as np


def load_data(file_path):
    """
    加载数据
    :param file_path: 数据目录
    :return: DataInput对象
    """
    with open(file_path, mode='rb') as file:
        obj = pickle.load(file)

    return obj


def save_data(obj, file_path):
    """
    保存数据
    :param obj: DataInput对象
    :param file_path: 保存路径
    """
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    with open(file=file_path, mode='wb') as file:
        pickle.dump(obj, file=file)


def load_whale_labels(path):
    """加载CSV文件"""
    labels = []
    reader = csv.reader(open(path, encoding='utf-8'))

    for pair in reader:
        file = str(pair[0])
        if file.startswith('train'):
            value = int(pair[1])
            labels.append(value)

    return labels


def split_dataset(pairs, valid_percent=0.2, test_percent=0.2, num_classes=2, ret_indices=False):
    """Split dataset into balanced train、valid and test set."""
    assert test_percent + valid_percent <= 1

    labels = pairs['label']
    train_indices = []
    valid_indices = []
    test_indices = []

    for i in range(num_classes):
        label_indices = np.squeeze(np.where(labels == i))

        valid_num = int(len(label_indices) * valid_percent)
        test_num = int(len(label_indices) * test_percent)

        tests = np.random.choice(label_indices, test_num, replace=False)
        remains = np.setdiff1d(label_indices, tests)
        valids = np.random.choice(remains, valid_num, replace=False)
        trains = np.setdiff1d(remains, valids)

        train_indices.extend(trains)
        valid_indices.extend(valids)
        test_indices.extend(tests)

    if ret_indices:
        return train_indices, valid_indices, test_indices
    else:
        return pairs[train_indices], pairs[valid_indices], pairs[test_indices]


def augmented_samples(feature, num_augs):
    """Do data augmentation for audio feature."""
    samples = []

    return


if __name__ == '__main__':
    # x = np.ones(shape=(100, 200))
    # y = np.zeros(shape=(100, 1))
    # size = np.alen(x)
    # data_input = DataInput(data=x, label=y, size=size)
    # # save_data(obj=data_input,
    # #           file_path='data/data.bin')
    #
    # data = load_data('data/data.bin')
    # print(data.size)

    pairs = load_data('../data/whale/pairs')

    trains, valids, tests = split_dataset(pairs, 0.2)
