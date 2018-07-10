import abc
import argparse
import glob
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.utils import to_categorical

import hparams
import utils.misc_utils as utils
from models_keras.base_model import logger
from models_keras.vggish_model import VggishModel
from strategies import al_metrics
from utils import data_utils
from utils.data_container import DataContainer

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)

session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(session)


def init_stats(**kwargs):
    if kwargs:
        stats_df = pd.DataFrame(
            data={"round": [], "num_new_labeled": [], "num_pseudo_labeled": [],
                  "num_total_labeled": [], "eval_acc": [], "eval_pre": [], "eval_recall": [],
                  "eval_f1": [], "pseudo_acc": [], **kwargs})
    else:
        stats_df = pd.DataFrame(
            data={"round": [], "num_new_labeled": [], "num_pseudo_labeled": [],
                  "num_total_labeled": [], "eval_acc": [], "eval_pre": [], "eval_recall": [],
                  "eval_f1": [], "pseudo_acc": []})

    return stats_df


def add_stats(stats_df, al_round, num_total_labeled, eval_res,
              num_new_labeled=None, num_pseudo_labeled=None, pseudo_acc=None,
              **kwargs):
    """
    Add stats of current round to 'stats_df'.
    :param stats_df:
    :param al_round:
    :param num_new_labeled:
    :param num_pseudo_labeled:
    :param num_total_labeled:
    :param eval_res: loss, accuracy, precision, recall, f1score
    :param pseudo_acc:
    :return:
    """
    assert isinstance(stats_df, pd.DataFrame) and len(eval_res) == 5

    if kwargs:
        new_df = pd.DataFrame(
            data={"round": [al_round], "num_new_labeled": [num_new_labeled],
                  "num_pseudo_labeled": [num_pseudo_labeled],
                  "num_total_labeled": [num_total_labeled],
                  "eval_acc": [eval_res[1]], "eval_pre": [eval_res[2]],
                  "eval_recall": [eval_res[3]], "eval_f1": [eval_res[4]],
                  "pseudo_acc": [pseudo_acc], **kwargs})
    else:
        new_df = pd.DataFrame(
            data={"round": [al_round], "num_new_labeled": [num_new_labeled],
                  "num_pseudo_labeled": [num_pseudo_labeled],
                  "num_total_labeled": [num_total_labeled],
                  "eval_acc": [eval_res[1]], "eval_pre": [eval_res[2]],
                  "eval_recall": [eval_res[3]], "eval_f1": [eval_res[4]],
                  "pseudo_acc": [pseudo_acc]})

    logger.info("Stats updated:\n %s" % new_df.to_string())

    return stats_df.append(new_df)


def load_stats(path):
    if path and os.path.exists(path) and os.path.isfile(path):
        stats_df = pd.read_csv(path)
    else:
        stats_df = None

    return stats_df


def save_stats(path, stats_df):
    utils.ensure_path_exist(path)
    stats_df.to_csv(path, index=False)


class Framework:

    def __init__(self, data_container, model, test_features, test_labels,
                 stats_path=None, num_select_per_round=200, rounds_per_stats=1,
                 max_round=50, pre_train=True, using_hist=True, **al_args):
        self.data_container = data_container
        self.model = model
        self.test_features = test_features
        self.test_labels = test_labels
        self.stats_path = stats_path
        self.num_select_per_round = num_select_per_round
        self.rounds_per_stats = rounds_per_stats
        self.max_round = max_round
        self.pre_train = pre_train
        self.al_args = al_args
        self.using_hist = using_hist
        self._round = 0
        self._stats_df = self._init_stats()
        self._init_model()

    def _init_stats(self):
        # Init stats.
        utils.ensure_path_exist(self.stats_path)
        stats_df = load_stats(self.stats_path)
        if stats_df is None:
            stats_df = init_stats(**self.al_args)
        print("Stats inited: \n%s" % stats_df.to_string())

        return stats_df

    def _init_model(self):
        # Restore data and .
        self._round = self._stats_df['round'].iloc[-1] + 1 if self._stats_df.shape[0] else 0

        if self._round > 0:
            # Restore data structure and continue the active learning process.
            self.data_container = data_utils.load_data(
                os.path.join(os.path.dirname(self.stats_path), 'data_container.np'))
        else:
            if self.pre_train:
                # # Train from scratch.
                logger.info('Start pre-training.')
                eval_res = self._train_and_eval()
            else:
                logger.info('Start pre-eval.')
                eval_res = self._train_and_eval(train=False)

            self._stats_df = add_stats(self._stats_df, 0, num_new_labeled=None,
                                       num_pseudo_labeled=None,
                                       num_total_labeled=self.data_container.labeled_data.shape[0],
                                       eval_res=eval_res, pseudo_acc=None,
                                       **self.al_args)
            self._round = 1
            # logger.info('Eval result: %s' % eval_res)

    def _train_and_eval(self, train=True, eval=True):
        if train:
            self.model.train(self.data_container.labeled_features,
                             self.data_container.labeled_labels,
                             self.test_features, self.test_labels)
        eval_res = None
        if eval:
            # Evaluation for the model trained.
            self.model.reload_weights()
            eval_res = self.model.evaluate(self.test_features, self.test_labels)
            logger.info('# Round %d, num_new_labeled: %d, eva_res: %s.' %
                        (0, self.data_container.labeled_data.shape[0], eval_res))

        return eval_res

    @abc.abstractmethod
    def active_select(self):
        """
        Do active selection by different strategies.
        :rtype al_metrics.ActiveSample
        """
        pass

    def make_iter_trainset(self,
                           new_labeled_indices,
                           pseudo_labeled_indices=None,
                           pseudo_labels=None):
        """
        Make train set of current iteration.
        :rtype (features, labels)
        """
        if self.using_hist:
            cur_train_features, cur_train_labels = self.data_container.labeled_features, \
                                                   self.data_container.labeled_labels
        else:
            cur_train_features, cur_train_labels = [], []

        if len(cur_train_features) > 0 and len(cur_train_labels) > 0:
            cur_train_features = np.concatenate(
                [cur_train_features, self.data_container.unlabeled_features[new_labeled_indices]])
            cur_train_labels = np.concatenate(
                [cur_train_labels, self.data_container.unlabeled_labels[new_labeled_indices]])
        else:
            cur_train_features = self.data_container.unlabeled_features[new_labeled_indices]
            cur_train_labels = self.data_container.unlabeled_labels[new_labeled_indices]

        return cur_train_features, cur_train_labels

    def run(self):
        logger.info('# Running active learning process.')
        start_time = time.time()

        # Run active learning process until all data is labeled.
        while self._round <= self.max_round and self.data_container.unlabeled_data.shape[0]:
            logger.info("\n%s Round %d %s" % ("#" * 40, self._round, "#" * 40))

            # Do active selection.
            al_res = self.active_select()

            # Collect data and stats.
            # new_labeled_data = data_container.unlabeled_data[al_res.selected]
            pseudo_labeled_data = self.data_container.unlabeled_data[al_res.pseudo_labeled]
            num_new_labeled = np.alen(al_res.selected) if al_res.selected is not None else 0
            num_pseudo_labeled = np.alen(al_res.pseudo_labeled) if al_res.pseudo_labeled is not None else 0
            pseudo_acc = None if num_pseudo_labeled == 0 else np.sum(np.equal(
                al_res.pseudo_labels, pseudo_labeled_data['label'].tolist())) / num_pseudo_labeled

            # TODO Just test.
            if num_new_labeled > 0:
                new_labeled_indices = al_res.selected
            else:
                num = 200
                indices = [i for i in range(self.data_container.unlabeled_data.shape[0])]
                logger.info("No new labeled data, %d samples was random selected." % num)
                new_labeled_indices = indices if len(indices) <= 200 else \
                    np.random.choice(indices, size=num, replace=False)

            cur_train_features, cur_train_labels = self.make_iter_trainset(
                new_labeled_indices, al_res.pseudo_labeled, al_res.pseudo_labels)
            # Update cache.
            self.data_container.update(new_labeled_idices=new_labeled_indices)

            self.model.train(cur_train_features, cur_train_labels,
                             self.test_features, self.test_labels)

            # Fine-tuning with new labeled and pseudo labeled data.
            eval_res = self._train_and_eval(train=False)
            self._stats_df = add_stats(self._stats_df, self._round, num_new_labeled=num_new_labeled,
                                       num_pseudo_labeled=num_pseudo_labeled,
                                       num_total_labeled=self.data_container.labeled_data.shape[0],
                                       eval_res=eval_res, pseudo_acc=pseudo_acc, **self.al_args)

            if self._round % self.rounds_per_stats == 0:
                # Over write saved stats every 3 times.
                save_stats(self.stats_path, self._stats_df)
                self.data_container.to_file(
                    os.path.join(os.path.dirname(self.stats_path), 'data_container.np'))

            self._round += 1

        # Save stats.
        save_stats(self.stats_path, self._stats_df)
        self.data_container.to_file(
            os.path.join(os.path.dirname(self.stats_path), 'data_container.np'))

        logger.info("# Done, time: %s. Stats:\n%s" %
                    (str(time.time() - start_time), self._stats_df.to_string()))


class EntropyFramework(Framework):

    def __init__(self, data_container, model, test_features, test_labels,
                 stats_path=None, num_select_per_round=200, rounds_per_stats=1,
                 max_round=50, pre_train=True, using_hist=True,
                 threshold=0.047, decay_rate=0.001):
        super(EntropyFramework, self).__init__(
            data_container, model, test_features, test_labels,
            stats_path, num_select_per_round, rounds_per_stats, max_round,
            pre_train, using_hist, threshold=threshold, decay_rate=decay_rate)

        if 'threshold' not in self.al_args or 'decay_rate' not in self.al_args:
            raise ValueError('Invalid `**kwargs` param.')

    def active_select(self):
        # Active selection.
        al_res = al_metrics.entropy(
            model=self.model,
            features=self.data_container.unlabeled_data['feature'],
            num_select=self.num_select_per_round,
            alpha=self.al_args['threshold'])

        # Decay threshold.
        self.al_args['threshold'] = self.al_args['threshold'] - self.al_args['decay_rate'] * (self._round - 1)
        if self.al_args['threshold'] < 0.0001:
            self.al_args['threshold'] = 0.0001

        return al_res


class EntropyPLFramework(Framework):

    def __init__(self, data_container, model, test_features, test_labels,
                 stats_path=None, num_select_per_round=200, rounds_per_stats=1,
                 max_round=50, pre_train=True, using_hist=True, threshold=0.047,
                 decay_rate=0.001):
        super(EntropyPLFramework, self).__init__(
            data_container, model, test_features, test_labels,
            stats_path, num_select_per_round, rounds_per_stats, max_round,
            pre_train, using_hist, threshold=threshold, decay_rate=decay_rate)

        if 'threshold' not in self.al_args or 'decay_rate' not in self.al_args:
            raise ValueError('Invalid `**kwargs` param.')

    def make_iter_trainset(self,
                           new_labeled_indices,
                           pseudo_labeled_indices=None,
                           pseudo_labels=None):
        # 加入伪标记
        if self.using_hist:
            cur_train_features, cur_train_labels = self.data_container.labeled_features, \
                                                   self.data_container.labeled_labels
        else:
            cur_train_features, cur_train_labels = [], []

        if len(cur_train_features) > 0 and len(cur_train_labels) > 0:
            cur_train_features = np.concatenate(
                [cur_train_features, self.data_container.unlabeled_features[new_labeled_indices],
                 self.data_container.unlabeled_features[pseudo_labeled_indices]])
            cur_train_labels = np.concatenate(
                [cur_train_labels, self.data_container.unlabeled_labels[new_labeled_indices],
                 to_categorical(pseudo_labels)])
        else:
            cur_train_features = np.concatenate(
                [self.data_container.unlabeled_features[new_labeled_indices],
                 self.data_container.unlabeled_features[pseudo_labeled_indices]])
            cur_train_labels = np.concatenate(
                [self.data_container.unlabeled_labels[new_labeled_indices],
                 to_categorical(pseudo_labels)])

        return cur_train_features, cur_train_labels

    def active_select(self):
        # Active selection.
        al_res = al_metrics.entropy(
            model=self.model,
            features=self.data_container.unlabeled_data['feature'],
            num_select=self.num_select_per_round,
            alpha=self.al_args['threshold'])

        # Decay threshold.
        self.al_args['threshold'] = self.al_args['threshold'] - self.al_args['decay_rate'] * (self._round - 1)
        if self.al_args['threshold'] < 0.0001:
            self.al_args['threshold'] = 0.0001

        return al_res


class EGLFramework(Framework):

    def active_select(self):
        # Active selection.
        al_res = al_metrics.egl(
            model=self.model,
            data_container=self.data_container,
            num_select=self.num_select_per_round)

        return al_res


class DPCFramework(Framework):

    def active_select(self):
        # Active selection.
        al_res = al_metrics.density_peak_halos(
            model=self.model,
            data_container=self.data_container,
            num_select=self.num_select_per_round)

        return al_res


class EDPCFramework(Framework):
    SEL_HIST_THRESHOLD = 3000

    def __init__(self, data_container, model, test_features, test_labels,
                 stats_path=None, num_select_per_round=200, rounds_per_stats=1,
                 max_round=50, pre_train=True, threshold=0.047, decay_rate=0.001,
                 using_hist=True):
        super(EDPCFramework, self).__init__(
            data_container, model, test_features, test_labels,
            stats_path, num_select_per_round, rounds_per_stats, max_round,
            pre_train, using_hist, threshold=threshold, decay_rate=decay_rate)

        if 'threshold' not in self.al_args or 'decay_rate' not in self.al_args:
            raise ValueError('Invalid `**kwargs` param.')

    def make_iter_trainset(self,
                           new_labeled_indices,
                           pseudo_labeled_indices=None,
                           pseudo_labels=None):
        # 挑选历史样本和当前新标记的样本作为当前批次的训练样本
        # _labeled_embeddings = self.model.embedding(self.data_container.labeled_features)
        # _select_features, _select_labels = self.data_container.filtered_labeled_samples(_labeled_embeddings)
        if self.using_hist:
            if len(self.data_container.labeled_labels) > self.SEL_HIST_THRESHOLD:
                logger.info("# Labeled samples reach threshold %d, start select historical samples." %
                            self.SEL_HIST_THRESHOLD)
                cur_train_features, cur_train_labels = self.data_container.filtered_labeled_samples2(
                    self.model, 0.5)
            else:
                logger.info("# Labeled samples not reach threshold %d, use all historical samples." %
                            self.SEL_HIST_THRESHOLD)
                cur_train_features, cur_train_labels = self.data_container.labeled_features, \
                                                       self.data_container.labeled_labels
        else:
            logger.info("# Do not use historical samples.")
            cur_train_features, cur_train_labels = [], []

        if len(cur_train_features) > 0 and len(cur_train_labels) > 0:
            cur_train_features = np.concatenate(
                [cur_train_features, self.data_container.unlabeled_features[new_labeled_indices]])
            cur_train_labels = np.concatenate(
                [cur_train_labels, self.data_container.unlabeled_labels[new_labeled_indices]])
        else:
            cur_train_features = self.data_container.unlabeled_features[new_labeled_indices]
            cur_train_labels = self.data_container.unlabeled_labels[new_labeled_indices]

        return cur_train_features, cur_train_labels

    def active_select(self):
        # Active selection.
        al_res = al_metrics.dpc_entropy_fusion(
            model=self.model,
            data_container=self.data_container,
            num_select=self.num_select_per_round)

        return al_res


class RandomFramework(Framework):

    def active_select(self):
        # Active selection.
        al_res = al_metrics.random(
            num_total=len(self.data_container.unlabeled_data['feature']),
            num_select=self.num_select_per_round)

        return al_res


def prepare_for_next_iteration(init_model_path, cur_work_space):
    assert os.path.exists(init_model_path) and os.path.isfile(init_model_path)

    if not os.path.exists(cur_work_space):
        os.makedirs(cur_work_space)

    # reset init model
    cur_model_path = os.path.join('%s/ckpt/vggish_model.h5' % cur_work_space)
    if os.path.exists(cur_model_path):
        os.remove(cur_model_path)

    if not os.path.exists(os.path.dirname(cur_model_path)):
        os.makedirs(os.path.dirname(cur_model_path))
    shutil.copy(init_model_path, os.path.dirname(cur_model_path))

    # move stats
    existing_files = glob.glob('%s/al_stats_[0-9].csv' % cur_work_space)
    dest_stats_path = os.path.join(cur_work_space, 'al_stats_%d.csv' % len(existing_files))
    stats_path = '%s/al_stats_temp.csv' % cur_work_space
    if os.path.exists(stats_path):
        shutil.move(stats_path, dest_stats_path)

    # remove data_container.np
    data_container_path = os.path.join(cur_work_space, 'data_container.np')
    if os.path.exists(data_container_path):
        os.remove(data_container_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add arguments.
    hparams.add_arguments(parser)

    # Acquire param file path.
    flags, unused = parser.parse_known_args()
    param_file = flags.param_file

    if not param_file or not os.path.exists(param_file):
        param_file = 'params/framework_hparams.json'

    print('Using param file `%s`' % param_file)

    # Read json config file.
    json_params = json.load(open(param_file))
    param_namespace = argparse.Namespace(**json_params)

    # Parse args.
    flags, unused = parser.parse_known_args(namespace=param_namespace)

    _input_shape = [60, 41, 2]
    _sub_dir = flags.sub_dir
    _num_classes = flags.num_classes
    _num_total = flags.num_total
    _sel_thresholds = flags.sel_thresholds

    # Framework object
    if flags.framework_type == 'dpc':
        framework_creator = DPCFramework
    elif flags.framework_type == 'edpc':
        framework_creator = EDPCFramework
    elif flags.framework_type == 'egl':
        framework_creator = EGLFramework
    elif flags.framework_type == 'entropy':
        framework_creator = EntropyFramework
    elif flags.framework_type == 'entropy_pl' or \
            flags.framework_type == 'hist_select':
        framework_creator = EDPCFramework
    else:
        framework_creator = RandomFramework

    print('Framework type: `%s`' % flags.framework_type)
    print('Framework architecture: `%s`, hist_sel_mode: %s' %
          (framework_creator.__name__, flags.hist_sel_mode))

    tf.set_random_seed(flags.random_seed)
    np.random.seed(flags.random_seed)

    # Load dataset.
    _pairs = data_utils.load_data('data/%s/pairs' % _sub_dir)
    _pairs = np.random.choice(_pairs, _num_total, replace=False)

    # Extract features and labels.
    _train_pairs, _, test_pairs = data_utils.split_dataset(
        pairs=_pairs, valid_percent=flags.valid_percent,
        test_percent=flags.test_percent, num_classes=flags.num_classes)

    _labels = np.array(_pairs['label'].tolist())
    _class_weights = [_num_total / sum(_labels == i) for i in range(_num_classes)]

    _train_features = np.reshape(_train_pairs['feature'].tolist(), [-1] + _input_shape)
    _train_labels = to_categorical(_train_pairs['label'].tolist())
    _test_features = np.reshape(test_pairs['feature'].tolist(), [-1] + _input_shape)
    _test_labels = to_categorical(test_pairs['label'].tolist())

    # Organize stats dirs
    init_model_path = 'outputs/%s/init_ckpt/vggish_model.h5' % flags.sub_dir
    cur_work_space = 'outputs/%s/stats/%s' % (flags.sub_dir, flags.framework_type)

    if flags.framework_type == 'hist_select':
        cur_work_space += '/%s' % flags.hist_sel_mode
        if flags.hist_sel_mode == 'certain':
            cur_work_space += '_%d' % flags.sel_thresholds[0]

    # Split into groups
    groups = glob.glob('%s/group_*' % cur_work_space)
    cur_work_space += '/group_%d' % 0

    prepare_for_next_iteration(init_model_path, cur_work_space)

    for sel_threshold in _sel_thresholds:
        # Label smoothing
        # smoothing_rate = 0.1
        # _train_labels = (1 - smoothing_rate) * _train_labels + smoothing_rate / 10
        # _test_labels = (1 - smoothing_rate) * _test_labels + smoothing_rate / 10

        # Build model
        _keras_model = VggishModel(
            class_weights=_class_weights, input_shape=_input_shape,
            num_classes=_num_classes, batch_size=flags.batch_size,
            learning_rate=flags.learning_rate, metric_baseline=flags.metric_baseline,
            num_epochs=flags.num_epochs, load_pretrained=flags.load_pretrained, feature_type='raw',
            output_dir='outputs/%s' % _sub_dir,
            model_dir='%s/ckpt' % cur_work_space)

        # _keras_model.train(_train_features, _train_labels,
        #                    _test_features, _test_labels)

        if flags.framework_type == 'entropy' or \
                flags.framework_type == 'edpc' or \
                flags.framework_type == 'hist_select':
            _embeddings = _keras_model.embedding(
                np.reshape(_pairs['feature'].tolist(), [-1] + _input_shape))
        else:
            _embeddings = None

        # Reset weights params.
        _keras_model.reload_weights()

        # Build container.
        _container = DataContainer(
            data=_train_pairs,
            labeled_percent=flags.labeled_percent,
            num_classes=_num_classes,
            feature_shape=_keras_model.input_shape,
            label_shape=_keras_model.output_shape,
            embeddings=_embeddings)

        print(_keras_model.get_model_summary())

        # Create framework object.
        framework = framework_creator(
            data_container=_container,
            model=_keras_model,
            test_features=_test_features,
            test_labels=_test_labels,
            stats_path='%s/al_stats_temp.csv' % cur_work_space,
            max_round=flags.max_round,
            num_select_per_round=flags.num_select_per_round,
            pre_train=flags.pre_train,
            threshold=flags.decay_threshold,
            decay_rate=flags.decay_rate,
            using_hist=flags.hist_sel_mode != 'no')

        # Set historical sample selection threshold.
        framework.SEL_HIST_THRESHOLD = sel_threshold

        # Run framework
        framework.run()

        # Do post process.
        prepare_for_next_iteration(init_model_path, cur_work_space)
