import tensorflow as tf

from utils import data_utils


class DataSetIterator:
    """Implement a data pipeline."""

    def __init__(self,
                 features,
                 labels,
                 batch_size=32,
                 shuffle=True,
                 skip_count=None,
                 random_seed=123,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 reshuffle_each_iteration=True):
        """
        创建一个数据迭代器
        :param features: 数据特征tensor
        :param labels: 数据标签tensor
        :param batch_size: 数据批大小
        :param shuffle: 是否将数据混洗
        :param skip_count: 跳过数据集开始skip_count个数据，如值-1则跳过整个数据集
        :param random_seed: 随机种子
        :param num_parallel_calls: 数据并行处理的数量，如未被设置，数据将被顺序处理
        :param output_buffer_size: 数据集最大缓冲数，限制输出数据集的大小
        :param reshuffle_each_iteration: 是否每次迭代都重新混洗数据
        """
        self._feature_dataset = tf.data.Dataset.from_tensor_slices(features)
        self._label_dataset = tf.data.Dataset.from_tensor_slices(labels)
        self._batch_size = batch_size
        self._skip_count = skip_count
        self._random_seed = random_seed
        self._num_parallel_calls = num_parallel_calls
        self._output_buffer_size = output_buffer_size
        self._shuffle = shuffle
        self._reshuffle_each_iteration = reshuffle_each_iteration

        # The following fields will be initialized by _init_iterator.
        self.initializer = None
        self.batched_features = None
        self.batched_labels = None

        # Preprocess
        self._init_iterator()

    def _init_iterator(self):
        """初始化数据集迭代器"""
        # Initialize self._output_buffer_size
        if not self._output_buffer_size:
            self._output_buffer_size = self._batch_size * 1000

        # Zip feature dataset and label dataset
        fl_dataset = tf.data.Dataset.zip(
            (self._feature_dataset, self._label_dataset))

        # Skip some examples.
        if self._skip_count is not None:
            fl_dataset.skip(self._skip_count)

        # Shuffle dataset
        if self._shuffle:
            fl_dataset = fl_dataset.shuffle(
                self._output_buffer_size, self._random_seed, self._reshuffle_each_iteration)

        # Batch dataset
        batched_dataset = fl_dataset.batch(self._batch_size)

        # Create an iterator
        iterator = batched_dataset.make_initializable_iterator()

        # Initialize fields.
        self.initializer = iterator.initializer
        next_batch = iterator.get_next()
        self.batched_features = next_batch[0]
        self.batched_labels = next_batch[1]


if __name__ == '__main__':
    _features = data_utils.load_data('../data/features')
    _labels = data_utils.load_data('../data/labels')

    _iterator = DataSetIterator(
        tf.constant(_features), tf.constant(_labels),
        batch_size=10, shuffle=False)

    sess = tf.Session()
    sess.run(_iterator.initializer)
    while True:
        batched_labels = sess.run(_iterator.batched_labels)
        print(batched_labels)
