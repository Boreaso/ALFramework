#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics.pairwise as pw
from sklearn import manifold

from utils.plot_utils import plot_scatter_diagram

logger = logging.getLogger("density_peaks_cluster")


def load_paperdata(distance_f):
    """
    Load distance from data

    Args:
            distance_f : distance file, the format is column1-index 1, column2-index 2, column3-distance

    Returns:
        distances dict, max distance, min distance, max continues id
    """
    logger.info("PROGRESS: load data")
    distances = {}
    min_dis, max_dis = sys.float_info.max, 0.0
    max_id = 0
    with open(distance_f, 'r') as fp:
        for line in fp:
            x1, x2, d = line.strip().split(' ')
            x1, x2 = int(x1), int(x2)
            max_id = max(max_id, x1, x2)
            dis = float(d)
            min_dis, max_dis = min(min_dis, dis), max(max_dis, dis)
            distances[(x1, x2)] = float(d)
            distances[(x2, x1)] = float(d)
    for i in range(max_id):
        distances[(i, i)] = 0.0
    logger.info("PROGRESS: load end")
    return distances, max_dis, min_dis, max_id


def select_dc(distance_matrix, percent=2.0, auto=False):
    """
    Select the local density threshold, default is the method used in paper, auto is `autoselect_dc`
    Returns:
        dc that local density threshold
    """
    logger.info("PROGRESS: select dc")

    if auto:
        return auto_select_dc(distance_matrix)

    num_points = distance_matrix.shape[0]
    position = int(num_points * (num_points - 1) / 2 * percent / 100)
    dc = sorted(distance_matrix.reshape(-1))[position * 2 + num_points]
    logger.info("PROGRESS: dc - " + str(dc))
    return dc


def auto_select_dc(distance_matrix):
    """
    Auto select the local density threshold that let average neighbor is 1-2 percent of all nodes.

    Returns:
        dc that local density threshold
    """
    num_points = distance_matrix.shape[0]
    max_dis = np.max(distance_matrix)
    min_dis = np.min(distance_matrix)

    dc = (max_dis + min_dis) / 2
    while True:
        nneighs = np.sum(distance_matrix < dc) / num_points ** 2
        if 0.01 <= nneighs <= 0.02:
            break
        # binary search
        if nneighs < 0.01:
            min_dis = dc
        else:
            max_dis = dc
        dc = (max_dis + min_dis) / 2
        if max_dis - min_dis < 0.0001:
            break

    logger.info("PROGRESS: dc - " + str(dc))

    return dc


def local_density(distance_matrix, dc, gauss=True, cutoff=False):
    """
    Compute all points' local density

    Args:
            :param cutoff: use cutoff func or not(can't use together with gauss)
            :param gauss: use gauss func or not(can't use together with cutoff)
            :param distance_matrix: distance matrix
            :param dc:

    Returns:
        local density vector that index is the point index that start from 1
    """
    assert gauss ^ cutoff
    logger.info("PROGRESS: compute local density")
    gauss_func = lambda d_ij, d_c: math.exp(- (d_ij / d_c) ** 2)
    cutoff_func = lambda d_ij, d_c: 1 if d_ij < d_c else 0
    func = gauss_func if gauss else cutoff_func

    num_points = distance_matrix.shape[0]
    rho = [0] * num_points
    for i in range(num_points - 1):
        # if i % 1000 == 0:
        #     print("Iter %d, time %s" % (i, time.time()))
        for j in range(i + 1, num_points):
            rho[i] += func(distance_matrix[i][j], dc)
            rho[j] += func(distance_matrix[i][j], dc)
        if i % (num_points / 10) == 0:
            logger.info("PROGRESS: at index #%i" % i)
    return np.array(rho, np.float32)


def min_distance(distance_matrix, rho, sort_rho_idx):
    """
    Compute all points' min distance to the higher local density point(which is the nearest neighbor)

    Args:
        :param rho: local density vector that index is the point index that start from 1
        :param distance_matrix:
        :param sort_rho_idx:
    Returns:
        min_distance vector, nearest neighbor vector

    """
    logger.info("PROGRESS: compute min distance to nearest higher density neigh")
    max_dis = np.max(distance_matrix)
    num_points = distance_matrix.shape[0]
    delta, nneigh = [float(max_dis)] * len(rho), [0] * len(rho)
    delta[sort_rho_idx[0]] = -1.
    for i in range(1, num_points):
        for j in range(0, i):
            old_i, old_j = sort_rho_idx[i], sort_rho_idx[j]
            if distance_matrix[old_i][old_j] < delta[old_i]:
                delta[old_i] = distance_matrix[old_i][old_j]
                nneigh[old_i] = old_j
        if i % (num_points / 10) == 0:
            logger.info("PROGRESS: at index #%i" % i)
    delta[sort_rho_idx[0]] = max(delta)
    return np.array(delta, np.float32), np.array(nneigh, np.float32)


def compute_bord(distance_matrix, dc, rho, cluster_dict):
    # 初始化数组 bord_rho 为 0,每个 cluster 定义一个 bord_rho 值
    num_points = distance_matrix.shape[0]
    bord_rho = {i: 0 for i in set(cluster_dict.values())}
    # 获取每一个 cluster 中平均密度的一个界 bord_rho
    for i in range(num_points - 1):
        for j in range(i + 1, num_points):
            # 距离足够小但不属于同一个 cluster 的 i 和 j
            if cluster_dict[i] != cluster_dict[j] and \
                    distance_matrix[i, j] <= dc:
                # 取 i,j 两点的平均局部密度
                rho_avg = (rho[i] + rho[j]) / 2
                if rho_avg > bord_rho[cluster_dict[i]]:
                    bord_rho[cluster_dict[i]] = rho_avg
                if rho_avg > bord_rho[cluster_dict[j]]:
                    bord_rho[cluster_dict[j]] = rho_avg
    return bord_rho


class DensityPeaksCluster:

    def __init__(self, n_clusters, dist_func='euclidean', percent=2.0, auto_choose_dc=True):
        self.n_clusters = n_clusters
        self.percent = percent
        self.auto_choose_dc = auto_choose_dc
        self.cluster_dict = None
        self.cluster_center = None
        self.distances = None
        self.num_points = None
        self.dist_func = None
        self.distance_matrix = None
        self.rho = None
        self.delta = None
        self.nneigh = None
        self.labels = []
        self.bord_rho = []
        self.dc = None

        if dist_func == 'euclidean':
            self.dist_func = pw.euclidean_distances
        elif dist_func == 'cosine':
            self.dist_func = pw.cosine_distances
        elif dist_func == 'manhattan':
            self.dist_func = pw.manhattan_distances
        else:
            raise ValueError("Invalid distance metric.")

    def _choose_centers(self, rho, delta):
        assert self.n_clusters <= len(rho) - 1

        # Rescale rho/delta and choose centers.
        cluster_centers_idx = np.argsort(
            -(delta / max(delta)) * (rho / max(rho)))[:self.n_clusters]

        return cluster_centers_idx

    def local_density(self, distance_matrix, dc=None):
        """
        Just compute local density
        """
        if dc is None:
            dc = select_dc(distance_matrix, percent=self.percent, auto=self.auto_choose_dc)
            logger.info('dc')
        rho = local_density(distance_matrix, dc)
        return rho

    def halos(self, scale=1):
        """
        Get halos flags.
        :param scale: rescale `bord_rho` to get proper number of halos.
        :return: a list, 1 means halo point.
        """
        if not self.cluster_dict:
            raise InterruptedError('You should call `fit` function first.')

        num_points = self.distance_matrix.shape[0]

        if not self.bord_rho:
            self.bord_rho = compute_bord(
                distance_matrix=self.distance_matrix, dc=self.dc,
                rho=self.rho, cluster_dict=self.cluster_dict)

        # 1 表示对应下标的点是halo
        halos = [1 if self.rho[i] < (self.bord_rho[self.cluster_dict[i]] / scale) else 0
                 for i in range(num_points)]

        return halos

    def fit(self, x, dc=None):
        """
        Cluster the data.

        Args:
            x  : data
            dc : local density threshold, call select_dc if dc is None

        Returns:
            local density vector, min_distance vector, nearest neighbor vector
        """
        if self.distance_matrix is not None:
            distance_matrix = self.distance_matrix
        else:
            distance_matrix = np.array(self.dist_func(x, x))
        num_points = distance_matrix.shape[0]

        assert not (dc is not None and self.auto_choose_dc)
        if dc is None:
            dc = select_dc(distance_matrix, percent=self.percent, auto=self.auto_choose_dc)

        rho = self.local_density(distance_matrix, dc=dc)
        sort_rho_idx = np.argsort(-rho)

        # Compute delta and nneigh array.
        delta, nneigh = min_distance(distance_matrix, rho, sort_rho_idx=sort_rho_idx)
        logger.info("PROGRESS: start cluster")

        # Choose cluster center.
        cluster_dict, cluster_center = {}, {}  # cl/icl in cluster_dp.m
        candidate_centers = self._choose_centers(rho, delta)
        for cluster_idx, center_idx in enumerate(candidate_centers):
            cluster_dict[center_idx] = cluster_idx
            cluster_center[cluster_idx] = center_idx

        # Assign other points to a cluster.
        for i, sort_idx in enumerate(sort_rho_idx):
            if sort_idx in cluster_dict:
                continue
            if nneigh[sort_idx] in cluster_dict:
                cluster_dict[sort_idx] = cluster_dict[nneigh[sort_idx]]
            else:
                cluster_dict[sort_idx] = -1

            if i % (num_points / 10) == 0:
                logger.info("PROGRESS: at index #%i" % i)

        self.cluster_dict, self.cluster_center = cluster_dict, cluster_center
        self.distance_matrix = distance_matrix
        self.num_points = num_points
        self.labels = [self.cluster_dict[i] for i in range(num_points)]
        self.dc = dc
        self.rho = rho
        self.delta = delta
        self.nneigh = nneigh
        logger.info("PROGRESS: ended")


def plot_rho_delta(rho, delta, style_list=None):
    """
    Plot scatter diagram for rho-delta points

    Args:
            :param delta:
            :param rho:
            :param style_list:
    """
    logger.info("PLOT: rho-delta plot")
    plot_scatter_diagram(
        0, rho, delta, x_label='rho', y_label='delta',
        title='rho-delta', style_list=style_list)


def plot_cluster(cluster):
    """
    Plot scatter diagram for final points that using multi-dimensional scaling for data

    Args:
            cluster : DensityPeakCluster object
    """
    logger.info("PLOT: cluster result, start multi-dimensional scaling")

    # fo = open(r'./tmp.txt', 'w')
    # fo.write('\n'.join(map(str, cluster.labels)))
    # fo.close()

    mds = manifold.MDS(
        max_iter=200, eps=1e-4, n_init=1, n_jobs=4,
        dissimilarity='precomputed')
    dp_mds = mds.fit_transform(cluster.distance_matrix)

    print("num halos: %d" % sum(cluster.halos))
    for i in range(len(cluster.labels)):
        if cluster.halos[i] == 1:
            cluster.labels[i] = -1

    logger.info("PLOT: end mds, start plot")
    style_labels = [1 if i in cluster.cluster_center.values() else 0
                    for i in range(cluster.num_points)]
    plot_scatter_diagram(0, cluster.rho, cluster.delta,
                         x_label='rho', y_label='delta',
                         title='rho-delta', style_list=style_labels,
                         show=False)
    plot_scatter_diagram(1, dp_mds[:, 0], dp_mds[:, 1],
                         title='cluster', style_list=cluster.labels,
                         show=False)
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    dpc = DensityPeaksCluster(n_clusters=6, dist_func='cosine')

    _distances, _max_dis, _min_dis, _num_points = load_paperdata('../data/cluster/example_distances.dat')

    _distance_matrix = np.zeros(shape=(_num_points, _num_points))
    for key in _distances.keys():
        _distance_matrix[key[0] - 1, key[1] - 1] = _distances[key]

    # rho = dpcluster.local_density(distance_matrix)
    # sort_rho_idx = np.argsort(-rho)
    # delta, nneigh = min_distance(distance_matrix, rho, sort_rho_idx=sort_rho_idx)
    #
    # tmp_rho, tmp_delta = rho[1:], delta[1:]
    # cluster_centers = np.argsort(-(tmp_delta / max(tmp_delta)) * (tmp_rho / max(tmp_rho)))[:5]
    # density_threshold = min(tmp_delta[cluster_centers])
    # distance_threshold = min(tmp_rho[cluster_centers])
    #
    # style_list = np.array([0] * len(tmp_rho))
    # style_list[cluster_centers] = 1
    # plot_rho_delta(rho, delta, style_list)  # plot to choose the threthold

    arr = np.random.randn(200, 128)

    dpc.distance_matrix = _distance_matrix
    dpc.fit(x=arr)
    logger.info(str(len(dpc.cluster_center)) + ' center as below')
    for _idx, _center in dpc.cluster_center.items():
        logger.info('%d %f %f' % (_idx, dpc.rho[_center], dpc.delta[_center]))
    plot_cluster(dpc)
