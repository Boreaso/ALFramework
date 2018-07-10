import time

import numpy as np
from keras.utils import to_categorical
from sklearn import manifold

import utils.data_utils as data_utils
import utils.misc_utils as utils
from strategies.density_peaks_cluster import DensityPeaksCluster


class DataContainer:
    """Cache for labeled and unlabeled data during active learning process."""

    def __init__(self, data, labeled_percent, num_classes,
                 feature_shape, label_shape, embeddings=None,
                 use_embedding=False):

        assert isinstance(data, np.ndarray)

        # self._data = xar.Dataset({'feature': xar.DataArray(data[0]), 'label': data[1]})
        # self._data = pd.DataFrame(data={'id': [str(i) for i in range(len(data[0]))],
        #                                 'feature': data[0],
        #                                 'label': data[1]},
        #                           columns=['id', 'feature', 'label'])
        # self._data_l = self._data.iloc[labeled_indices]
        # self._data_u = self._data.iloc[unlabeled_indices]
        self._data = data

        unlabeled_indices, labeled_indices, _ = data_utils.split_dataset(
            data, labeled_percent, 0, num_classes=num_classes, ret_indices=True)
        self._data_l = data[labeled_indices]
        self._data_u = data[unlabeled_indices]

        self._num_classes = num_classes
        self._feature_shape = feature_shape
        self._label_shape = label_shape
        self._embeddings = np.array(embeddings) if embeddings is not None else None

        if self._embeddings is not None:
            self._embedded_data = self._embeddings
            self._embedded_data_u = self._embeddings[unlabeled_indices]
            self._embedded_data_l = self._embeddings[labeled_indices]
        elif use_embedding:
            print("Start embedding.")
            _tmp_data = np.array(data['feature'].tolist()).reshape(feature_shape)
            if np.ndim(data) == 4:
                _tmp_data = np.sum(_tmp_data, axis=-1)

            # Reshape.
            shape = [_tmp_data.shape[0], 1]
            for i in range(1, np.ndim(_tmp_data)):
                shape[1] *= _tmp_data.shape[i]
            _tmp_data = _tmp_data.reshape(shape)

            embedding_func = manifold.SpectralEmbedding(n_components=128)
            self._embedded_data = embedding_func.fit_transform(_tmp_data)
            self._embedded_data_l = self._embedded_data[unlabeled_indices]
            self._embedded_data_u = self._embedded_data[labeled_indices]

            print('Embedding finished, time %s' % time.time())

    @property
    def all_data(self):
        return self._data

    @property
    def labeled_data(self):
        return self._data_l

    @property
    def unlabeled_data(self):
        return self._data_u

    @property
    def labeled_features(self):
        return np.array(self._data_l['feature'].tolist()) \
            .reshape(self._feature_shape)

    @property
    def labeled_labels(self):
        labels = self._data_l['label'].tolist()
        labels = to_categorical(labels, num_classes=self._num_classes)
        return labels.reshape(self._label_shape)

    def filtered_labeled_samples(self, labeled_embeddings, select_rate=0.5):
        # 使用深度特征挑选最具代表性的历史样本
        labeled_embeddings = np.array(labeled_embeddings)
        select_indices = []
        for cls in range(self._num_classes):
            cur_indices = np.squeeze(np.argwhere(self._data_l['label'] == 1))
            num_clusters = int(len(cur_indices) * select_rate)
            dpc = DensityPeaksCluster(num_clusters, percent=0.2, auto_choose_dc=False)
            dpc.fit(labeled_embeddings[cur_indices])
            tmp_select_indices = cur_indices[list(dpc.cluster_center.values())]
            select_indices += list(tmp_select_indices)
        return self.labeled_features[select_indices], self.labeled_labels[select_indices]

    def filtered_labeled_samples2(self, model, percent=0.5):
        _preds, _entropys = model.get_entropy(self.labeled_features)
        _pseudo_labels = np.argmax(_preds, axis=-1)
        _ground_truth = np.argmax(self.labeled_labels, axis=-1)
        _correct_indices = np.argwhere(_pseudo_labels == _ground_truth)
        _select_indices = []
        for cls in range(self._num_classes):
            _cls_indices = np.argwhere(_ground_truth == cls)
            _cls_correct_indices = np.intersect1d(_correct_indices, _cls_indices)
            _cls_correct_entropys = _entropys[np.squeeze(_cls_correct_indices)]
            # _max_ent_indices = np.argsort(-_entropys)
            _min_ent_indices = np.squeeze(_cls_correct_indices[np.argsort(-_cls_correct_entropys)])

            _num_select = int(len(_min_ent_indices) * percent)
            _select_indices += list(_min_ent_indices[:_num_select])

        _select_features = self.labeled_features[_select_indices]
        _select_labels = self.labeled_labels[_select_indices]

        return _select_features, _select_labels

    @property
    def unlabeled_features(self):
        return np.array(self._data_u['feature'].tolist()) \
            .reshape(self._feature_shape)

    @property
    def labeled_embeddings(self):
        return self._embedded_data_l

    @property
    def unlabeled_embeddings(self):
        return self._embedded_data_u

    @property
    def unlabeled_labels(self):
        labels = self._data_u['label'].tolist()
        labels = to_categorical(labels, num_classes=self._num_classes)
        return labels.reshape(self._label_shape)

    def describe(self):
        print("# Labeled data description:")
        print("num: %d " % np.alen(self._data_l))
        print("# Unlabeled data description:")
        print("num: %d " % np.alen(self._data_u))

    def update(self, new_labeled_idices):
        """Update labeled and unlabeled data cache.Remove the specified items
        from unlabeled data cache and add them to labeled data cache."""

        # # Get selected bool values.
        # removed_from_u = self._data_u.id.isin(new_labeled_ids)
        #
        # # Update cache.
        # self._data_u = self._data_u[~removed_from_u]
        # self._data_u.reindex()
        #
        # self._data_l = self._data_l.append(self._data_u[removed_from_u])

        # Update cache.
        new_labeled_idices = np.squeeze(new_labeled_idices)
        self._data_l = np.concatenate([self._data_l, self._data_u[new_labeled_idices]])
        np.random.shuffle(self._data_l)  # shuffle new labeled data.
        self._data_u = np.delete(self._data_u, new_labeled_idices, axis=0)

        if self._embeddings is not None:
            # Update embedded data cache.
            self._embedded_data_l = np.concatenate(
                [self._embedded_data, self._embedded_data_u[new_labeled_idices]])
            np.random.shuffle(self._data_l)  # shuffle new labeled data.
            self._embedded_data_u = np.delete(
                self._embedded_data_u, new_labeled_idices, axis=0)

    def to_file(self, path):
        utils.ensure_path_exist(path)
        data_utils.save_data(self, file_path=path)


if __name__ == '__main__':
    _features = data_utils.load_data('../data/features')
    _labels = data_utils.load_data('../data/labels')

    pairs = [(f, l) for f, l in zip(_features, _labels)]

    data_utils.save_data(np.array(pairs, dtype=[('feature', np.ndarray), ('label', np.int)]), '../data/pairs')
