import collections
import time
from collections import Counter

import numpy as np

from framework import logger
from strategies.aift import compute_r_matrix
from strategies.density_peaks_cluster import DensityPeaksCluster


class ActiveSamples(
    collections.namedtuple('ActiveSamples',
                           ['selected',
                            'pseudo_labeled',
                            'pseudo_labels'])):
    """
    :param selected: Sample indices of which are selected by active learning algorithm.
    :param pseudo_labeled: Sample indices of which are pseudo labeled by current model.
    :param pseudo_labels:  Pseudo labels of rejected samples which has high confidence.
                          'None' means the sample doesn't reach a threshold.
    """
    pass


def random(num_total, num_select):
    """
    Random select num_select samples from num_total samples.
    :param num_total total number of samples.
    :param num_select select number of samples.
    """
    if num_select < num_total:
        _selected = np.random.choice(range(num_total), num_select, replace=False)
    else:
        _selected = [i for i in range(num_total)]

    return ActiveSamples(
        selected=np.squeeze(_selected),
        pseudo_labeled=[],
        pseudo_labels=[])


def ecg(model, data_container, num_select):
    """
    根据模型对于当前样本的梯度值选取样本
    :param data_container: 数据容器
    :param model: 当前模型
    :param num_select：选取样本的个数
    :return: ActiveSamples obj
    """
    features = data_container.unlabeled_features
    labels = np.array(data_container.all_data['label'].tolist())

    # Gradients of the second last layer.
    predictions, gradients = model.get_gradients(features)
    assert np.ndim(predictions) == 2 and np.ndim(gradients) == 3

    if predictions.shape[1] == 1:
        # Binary classification with 'sigmoid', insert probabilities of 0 and copy gradient of 1
        predictions = np.insert(predictions, 0, 1 - predictions[:, 0], axis=1)
        gradients = np.repeat(gradients, 2, axis=2)

    # Now 'predictions' has shape:(num_samples, num_classes),
    # 'gradients' has shape:(num_samples, num_units_pre, num_classes)
    # Compute ecg values. ecg = sum(p[j] * norm(gradients[i])),
    # i means sample i, j means class j.
    assert predictions.shape[1] == gradients.shape[2]
    num_samples = predictions.shape[0]

    g_norms = np.linalg.norm(gradients, axis=1)  # shape:(num_samples, num_classes)
    ecg_values = np.sum(predictions * g_norms, axis=1).reshape(num_samples)
    ecg_args = np.argsort(ecg_values, axis=0)

    return ActiveSamples(
        selected=ecg_args[-num_select:],
        pseudo_labeled=None,
        pseudo_labels=None)


def egl(model, data_container, num_select):
    features = data_container.unlabeled_features
    labels = np.array(data_container.all_data['label'].tolist())

    egl_values = model.get_egl(features)
    egl_args = np.argsort(egl_values, axis=0)

    return ActiveSamples(
        selected=egl_args[-num_select:],
        pseudo_labeled=None,
        pseudo_labels=None)


def entropy(model, features, num_select, alpha=0.045):
    """
    选取entropy < alpha 的样本作为确信样本，其余作为不确信样本
    :param features: 当前模型预测样本的熵
    :param model: 当前模型
    :param num_select:选取样本个数
    :param alpha: 选取阈值
    :return: ActiveSamples obj
    """
    if np.ndim(features) == 1:
        features = np.array(features.tolist()).reshape(model.input_shape)

    # _preds shape:(num_samples, num_classes)
    # _entropys shape: (num_samples,)
    _preds, _entropys = model.get_entropy(features)
    _selected = np.argwhere(_entropys >= alpha)

    # Select 'num_select' samples.
    if np.alen(_selected) > num_select:
        _entropys_selected = _entropys[np.squeeze(_selected)]
        _max_entropy_indices = np.argsort(_entropys_selected)[-num_select:]
        _selected = _selected[_max_entropy_indices]

    # High confidence samples.
    _pseudo_labeled = np.squeeze(np.argwhere(_entropys < alpha))
    _pseudo_label = np.argmax(_preds[_pseudo_labeled], axis=1) \
        if np.alen(_pseudo_labeled) > 0 else []

    print('Entropy finished，%d selected, %d pseudo labeled.' % (len(_selected), len(_pseudo_labeled)))

    return ActiveSamples(
        selected=np.squeeze(_selected),
        pseudo_labeled=np.squeeze(_pseudo_labeled),
        pseudo_labels=np.squeeze(_pseudo_label))


def density_peak_halos(model, data_container, num_select):
    cluster_labels = None
    features = data_container.unlabeled_features
    labels = np.array(data_container.all_data['label'].tolist())

    if len(features) <= num_select:
        selected = [i for i in range(len(features))]
    else:
        if np.ndim(features) == 1:
            features = np.array(features.tolist()).reshape(model.input_shape)

        embeddings = model.embedding(features)
        num_cluster = 200

        start = time.time()
        dpc = DensityPeaksCluster(num_cluster, percent=0.2, auto_choose_dc=False)
        # dpc.fit(features.reshape([-1, 60 * 41 * 2]))
        dpc.fit(embeddings)
        print('Cluster finished, time %s' % (time.time() - start))

        min_diff = len(features)
        halos = []
        for i in range(1, 21):
            tmp_halos = dpc.halos(scale=i)
            tmp_diff = abs(sum(tmp_halos) - num_select)
            if tmp_diff < min_diff:
                min_diff = tmp_diff
                halos = tmp_halos

        selected = np.argwhere(np.array(halos) == 1).squeeze()

        # Centers
        center_labels = [labels[int(i)] for i in dpc.cluster_center.values()]
        # Halos
        halo_labels = [dpc.cluster_dict[i] for i in selected]
        target_labels = labels[selected]
        # Fusion
        # fusion = np.random.choice([int(x) for x in dpc.cluster_center.values()], 150, replace=False)
        centers = np.array([int(x) for x in dpc.cluster_center.values()])
        # center_rhos = [dpc.rho[x] for x in centers]
        # fusion = centers[np.argsort(center_rhos, axis=0)[-150:]]
        fusion = np.concatenate([centers, selected])
        fusion = np.random.choice(list(set(fusion)), num_select, replace=False)
        fusion_labels = labels[fusion]

        print('Num halos: %d' % len(selected))

        print("Cluster stats:")
        # print("Halos cluster: ", Counter(halo_labels))
        print("Ground truth distribution: ", Counter(labels))
        print("Centers ground truth: ", Counter(center_labels))
        print("Halos ground truth: ", Counter(target_labels))
        print("Fusion ground truth: ", Counter(fusion_labels))

        # raise EnvironmentError()

        # mds = manifold.TSNE()
        # dp_mds = mds.fit_transform(np.array(embeddings)[selected])
        #
        # plot_scatter_diagram(1, dp_mds[:, 0], dp_mds[:, 1],
        #                      title='cluster', style_list=cluster_labels)

    return ActiveSamples(selected=fusion,
                         pseudo_labeled=None,
                         pseudo_labels=cluster_labels)


def dpc_entropy_fusion(model, data_container, num_select):
    features = data_container.unlabeled_features
    labels = np.array(data_container.all_data['label'].tolist())
    embeddings = data_container.unlabeled_embeddings
    # _preds shape:(num_samples, num_classes)
    # _entropys shape: (num_samples,)
    _preds, _entropys = model.get_entropy(features)

    # Select 'num_select' samples.
    if np.alen(_entropys) > num_select * 3:
        _max_entropy_indices = np.argsort(_entropys)[-num_select * 3:]
    else:
        _max_entropy_indices = np.array([i for i in range(len(_entropys))])

    logger.info('Max entropy distribution: %s' % Counter(labels[_max_entropy_indices]))

    if np.alen(_max_entropy_indices) > num_select:
        # Cluster
        # embeddings = model.embedding(features[_max_entropy_indices])
        dpc = DensityPeaksCluster(num_select, auto_choose_dc=True)
        dpc.fit(embeddings[_max_entropy_indices])

        # Max rho selection.
        # _max_rho_indices = np.argsort(dpc.rho, axis=0)[-num_select:]
        # _selected = _max_entropy_indices[_max_rho_indices]

        # Cluster centers selection.
        # _center_indices = [int(i) for i in dpc.cluster_center.values()]
        # _selected = _max_entropy_indices[_center_indices]

        # Max delta and centers selection
        _center_indices = [int(i) for i in dpc.cluster_center.values()]
        _center_indices = np.random.choice(_center_indices, 150, replace=False)
        _left_indices = np.array(list(set([i for i in range(len(_max_entropy_indices))]) - set(_center_indices)))
        _sorted_delta = np.argsort(-dpc.delta)  # Descending sort
        _max_delta_indices = list(set(_left_indices).intersection(set(_sorted_delta)))[:50]
        _selected = np.concatenate([_center_indices, _max_delta_indices])
        _selected = _max_entropy_indices[_selected]
    else:
        _selected = _max_entropy_indices

    # logger.info('DPC-Entropy distribution: %s' % Counter(labels[_center_indices]))
    logger.info('DPC-Entropy distribution: %s' % Counter(labels[_selected]))
    logger.info('DPC-Entropy finished，%d selected.' % len(_selected))

    return ActiveSamples(
        selected=np.squeeze(_selected),
        pseudo_labeled=None,
        pseudo_labels=None)


def aift_r_matrix(model, features, alpha):
    """Al strategy proposed by
    'http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_Fine-Tuning_Convolutional_Neural_CVPR_2017_paper.pdf'"""

    for feature in features:
        probs = model.predict(feature)
        compute_r_matrix(probs)
