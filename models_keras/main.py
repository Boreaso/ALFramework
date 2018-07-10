import os
import time

import numpy as np
import pandas as pd
from keras.utils import to_categorical

from models_keras.base_model import logger
from models_keras.vggish_model import VggishModel
from strategies import al_metrics
from utils import data_utils, misc_utils as utils
from utils.data_container import DataContainer


def init_stats():
    stats_df = pd.DataFrame(
        data={"round": [], "num_new_labeled": [], "num_pseudo_labeled": [],
              "num_total_labeled": [], "threshold": [], "decay_rate": [],
              "eval_acc": [], "eval_pre": [], "eval_recall": [], "eval_f1": [],
              "pseudo_acc": []})
    return stats_df


def reshape(array, shape):
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    if array.shape != shape:
        array = np.array(array.tolist()).reshape(shape)

    return array


def add_stats(stats_df, al_round, num_total_labeled, threshold, decay_rate, eval_res,
              num_new_labeled=None, num_pseudo_labeled=None, pseudo_acc=None):
    """
    Add stats of current round to 'stats_df'.
    :param decay_rate:
    :param threshold:
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

    new_df = pd.DataFrame(
        data={"round": [al_round], "num_new_labeled": [num_new_labeled],
              "num_pseudo_labeled": [num_pseudo_labeled],
              "num_total_labeled": [num_total_labeled],
              "threshold": [threshold],
              "decay_rate": [decay_rate],
              "eval_acc": [eval_res[1]], "eval_pre": [eval_res[2]],
              "eval_recall": [eval_res[3]], "eval_f1": [eval_res[4]],
              "pseudo_acc": [pseudo_acc]})

    logger.info("Stats updated:\n %s" % new_df.to_string())

    return stats_df.append(new_df)


def load_stats(path):
    if path and os.path.exists(path) and os.path.isfile(path):
        stats_df = data_utils.load_data(path)
    else:
        stats_df = None

    return stats_df


def save_stats(path, stats_df):
    utils.ensure_path_exist(path)
    data_utils.save_data(stats_df, file_path=path)


def run_entropy(data_container, model, test_features, test_lablels, stats_path=None,
                num_select_per_round=200, save_stats_per_round=1,
                max_round=50):
    # Init stats.
    utils.ensure_path_exist(stats_path)
    stats_df = load_stats(stats_path)
    if stats_df is None:
        stats_df = init_stats()
    print("Stats inited: \n%s" % stats_df.to_string())

    # Restore stats
    _round = stats_df['round'].iloc[-1] if stats_df.shape[0] else 0

    # if _round > 0:
    #     # Restore data structure and continue the active learning process.
    #     data_container = data_utils.load_data(
    #         os.path.join(os.path.dirname(stats_path), 'data_container.np'))
    # else:
    #     # # Train from scratch.
    #     logger.info('Start pre-training.')
    #     model.train(data_container.labeled_features,
    #                 data_container.labeled_labels,
    #                 test_features, test_lablels)
    #     # val_inputs=test_features,
    #     # val_labels=test_lablels)
    #
    #     # Evaluation for initial model.
    #     model.reload_weights()
    #     eval_res = model.evaluate(test_features, test_lablels)
    #     logger.info('# Round %d, num_new_labeled: %d, eva_res: %s.' %
    #                 (0, data_container.labeled_data.shape[0], eval_res))
    #     stats_df = add_stats(stats_df, 0, num_new_labeled=None,
    #                          num_pseudo_labeled=None,
    #                          num_total_labeled=data_container.labeled_data.shape[0],
    #                          eval_res=eval_res, pseudo_acc=None,
    #                          threshold=None, decay_rate=None)

    logger.info('# Active learning started, strategy: %s.' % "entropy")
    start_time = time.time()

    # Run active learning process until all data is labeled.
    _threshold = 0.045
    _decay_rate = 0.0001
    while _round <= max_round and data_container.unlabeled_data.shape[0]:
        logger.info("%s Round %d %s" % ("#" * 40, _round, "#" * 40))

        # Active selection.
        # al_res = al_metrics.entropy(
        #     model=model,
        #     features=data_container.unlabeled_data['feature'],
        #     num_select=num_select_per_round,
        #     alpha=_threshold)
        # # Decay threshold.
        # _threshold = _threshold - _decay_rate * (_round - 1)
        # if _threshold < 0.001:
        #     _threshold = 0.001
        al_res = al_metrics.density_peak_halos(
            model=model,
            features=data_container.unlabeled_data['feature'],
            num_select=num_select_per_round)

        # Collect data and stats.
        # new_labeled_data = data_container.unlabeled_data[al_res.selected]
        pseudo_labeled_data = data_container.unlabeled_data[al_res.pseudo_labeled]
        num_new_labeled = np.alen(al_res.selected)
        num_pseudo_labeled = np.alen(al_res.pseudo_labeled)
        pseudo_acc = None if num_pseudo_labeled == 0 else np.sum(np.equal(
            al_res.pseudo_labels, pseudo_labeled_data['label'].tolist())) / num_pseudo_labeled

        # TODO Just test.
        if num_new_labeled > 0:
            new_labeled_indices = al_res.selected
        else:
            num = 200
            inidices = [i for i in range(data_container.unlabeled_data.shape[0])]
            logger.info("No new labeled data, %d samples was random selected." % num)
            new_labeled_indices = inidices if len(inidices) <= 200 else \
                np.random.choice(inidices, size=num, replace=False)

        # Update cache.
        data_container.update(new_labeled_idices=new_labeled_indices)

        # Make train stuff. (combine new labeled and pseudo labeled data.)
        # train_data = new_labeled_data
        # pseudo_labeled_data['label'] = al_res.pseudo_labels
        # train_data = np.concatenate([train_data, pseudo_labeled_data])
        # train_feature = reshape(train_data['feature'], model.input_shape)
        # train_label = reshape(
        #     to_categorical(train_data['label'].tolist(), num_classes=2),
        #     model.output_shape)

        train_feature = data_container.labeled_features
        train_label = data_container.labeled_labels

        # Fine-tuning with new labeled and pseudo labeled data.
        model.train(train_feature, train_label,
                    val_inputs=test_features, val_labels=test_lablels)

        # Evaluation for this round.
        model.reload_weights()
        eval_res = model.evaluate(test_features, test_lablels)
        logger.info('# Round %d, num_new_labeled: %d, eva_res: %s.' %
                    (_round, data_container.labeled_data.shape[0], eval_res))
        stats_df = add_stats(stats_df, _round, num_new_labeled=num_new_labeled,
                             num_pseudo_labeled=num_pseudo_labeled,
                             num_total_labeled=data_container.labeled_data.shape[0],
                             eval_res=eval_res, pseudo_acc=pseudo_acc, threshold=_threshold,
                             decay_rate=_decay_rate)

        if _round % save_stats_per_round == 0:
            # Over write saved stats every 3 times.
            save_stats(stats_path, stats_df)
            data_container.to_file(
                os.path.join(os.path.dirname(stats_path), 'data_container.np'))

        _round += 1

    # Save stats.
    save_stats(stats_path, stats_df)
    data_container.to_file(
        os.path.join(os.path.dirname(stats_path), 'data_container.np'))

    logger.info("# Done, time: %s. Stats:\n%s" %
                (str(time.time() - start_time), stats_df.to_string()))


if __name__ == '__main__':
    np.random.seed(1)

    _input_shape = [60, 41, 2]
    _num_classes = 10
    _num_total = 12500

    _pairs = data_utils.load_data('../data/urbansound8k/pairs')
    _pairs = np.random.choice(_pairs, _num_total, replace=False)

    # Extract features and labels.
    _train_pairs, _, test_pairs = data_utils.split_dataset(
        pairs=_pairs, valid_percent=0, test_percent=0.2, num_classes=_num_classes)

    _labels = np.array(_pairs['label'].tolist())
    _class_weights = [_num_total / sum(_labels == i) for i in range(_num_classes)]

    _train_features = np.reshape(_train_pairs['feature'].tolist(), [-1] + _input_shape)
    _train_labels = to_categorical(_train_pairs['label'].tolist())
    _test_features = np.reshape(test_pairs['feature'].tolist(), [-1] + _input_shape)
    _test_labels = to_categorical(test_pairs['label'].tolist())

    # Build model
    _keras_model = VggishModel(
        class_weights=_class_weights, input_shape=_input_shape,
        num_classes=_num_classes, batch_size=64,
        learning_rate=0.0001, metric_baseline=0.5,
        num_epochs=30, load_pretrained=True, feature_type='raw',
        output_dir='outputs')

    # Build container.
    _container = DataContainer(
        data=_train_pairs,
        labeled_percent=0.1,
        num_classes=_num_classes,
        feature_shape=_keras_model.input_shape,
        label_shape=_keras_model.output_shape)

    print(_keras_model.get_model_summary())

    run_entropy(data_container=_container, model=_keras_model, test_features=_test_features,
                test_lablels=_test_labels, stats_path='outputs/stats/al_stats.pd', max_round=60)
