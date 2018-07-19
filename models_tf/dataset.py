import tensorflow as tf


class DatasetIterator:

    def __init__(self,
                 features,
                 targets,
                 batch_size,
                 shuffle=True,
                 random_seed=0):
        """
        数据迭代器。
        :param features: 特征
        :param targets: 标签
        :param batch_size: 批大小
        :param shuffle: 是否混洗
        :param random_seed: 随机种子
        """
        assert len(features) == len(targets)

        feature_dataset = tf.data.Dataset.from_tensor_slices(features)
        target_dataset = tf.data.Dataset.from_tensor_slices(targets)

        batched_dataset = tf.data.Dataset.zip((feature_dataset, target_dataset)).batch(batch_size)

        if shuffle:
            batched_dataset = batched_dataset.shuffle(
                buffer_size=batch_size * 1000, seed=random_seed,
                reshuffle_each_iteration=False)

        iterator = batched_dataset.make_initializable_iterator()
        next_batch = iterator.get_next()
        self.initializer = iterator.initializer
        self.batched_features = next_batch[0]
        self.batched_targets = next_batch[1]
