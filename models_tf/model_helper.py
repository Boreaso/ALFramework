import collections
import time

import tensorflow as tf
import tensorflow.contrib.training as tf_training

from utils import param_utils
from utils.iterator import DataSetIterator
from .base_model import ModeKeys

__all__ = ["get_initializer",
           "create_train_model"]


def get_initializer(init_op, seed=None, stddev=1.0, init_weight=None):
    """Create an initializer. init_weight is only for uniform."""
    if init_op == "uniform":
        assert init_weight
        return tf.random_uniform_initializer(
            -init_weight, init_weight, seed=seed)
    elif init_op == "truncated_normal":
        return tf.truncated_normal_initializer(
            stddev=stddev, seed=seed)
    elif init_op == "glorot_normal":
        return tf.keras.initializers.glorot_normal(
            seed=seed)
    elif init_op == "glorot_uniform":
        return tf.keras.initializers.glorot_uniform(
            seed=seed)
    elif init_op == "zeros":
        return tf.zeros_initializer()
    else:
        raise ValueError("Unknown init_op %s" % init_op)


class TrainModel(
    collections.namedtuple(
        "TrainModel",
        ("graph",
         "model",
         "iterator",
         "features_placeholder",
         "labels_placeholder",
         "skip_count_placeholder"))):
    pass


def create_train_model(hparams,
                       model_creator,
                       scope=None):
    """Create train graph, model, and iterator."""
    print("# Creating TrainModel...")

    batch_size = hparams.get('batch_size')
    shuffle = hparams.get('shuffle')

    graph = tf.Graph()
    with graph.as_default(), tf.container(scope or "train"):
        features_placeholder = tf.placeholder(shape=(None, None, None), dtype=tf.float32)
        labels_placeholder = tf.placeholder(shape=(None,), dtype=tf.int64)
        skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)

        iterator = DataSetIterator(
            features=features_placeholder,
            labels=labels_placeholder,
            skip_count=skip_count_placeholder,
            batch_size=batch_size,
            shuffle=shuffle)

        assert isinstance(hparams, tf_training.HParams)
        assert hparams.get('mode') in [ModeKeys.TRAIN, ModeKeys.TRAIN_CLASSIFIER, ModeKeys.FINE_TUNE]

        model_params = param_utils.get_model_params(hparams, iterator)
        model = model_creator(**model_params.values())

    return TrainModel(
        graph=graph,
        model=model,
        iterator=iterator,
        features_placeholder=features_placeholder,
        labels_placeholder=labels_placeholder,
        skip_count_placeholder=skip_count_placeholder)


class EmbeddingModel(
    collections.namedtuple(
        "EmbeddingModel",
        ("graph",
         "model",
         "iterator",
         "features_placeholder",
         "labels_placeholder"))):
    pass


def create_embedding_model(hparams,
                           model_creator,
                           scope=None):
    """Create embedding graph, model, and iterator."""
    print("# Creating EmbeddingModel...")

    batch_size = hparams.get('batch_size')
    shuffle = hparams.get('shuffle')

    graph = tf.Graph()
    with graph.as_default(), tf.container(scope or "embedding"):
        features_placeholder = tf.placeholder(shape=(None, None, None), dtype=tf.float32)
        labels_placeholder = tf.placeholder(shape=(None,), dtype=tf.int64)

        iterator = DataSetIterator(
            features=features_placeholder,
            labels=labels_placeholder,
            batch_size=batch_size,
            shuffle=shuffle)

        assert isinstance(hparams, tf_training.HParams)
        assert hparams.get('mode') == ModeKeys.EMBEDDING

        model_params = param_utils.get_model_params(hparams, iterator)
        model = model_creator(**model_params.values())

    return EmbeddingModel(
        graph=graph,
        model=model,
        iterator=iterator,
        features_placeholder=features_placeholder,
        labels_placeholder=labels_placeholder)


def load_model(model, ckpt, session, restore_mode, name):
    start_time = time.time()

    # Get the list of all currently existing variables
    if restore_mode == "all":
        model_vars = [v for v in session.graph.global_variables()]
    elif restore_mode == "embedding":
        model_vars = model.embedding_vars
    elif restore_mode == "classifier":
        model_vars = model.classifier_vars
    else:
        raise ValueError("Unknown restore_mode value.")

    # Use a Saver to restore just the variables selected above.
    saver = tf.train.Saver(model_vars, name='load_pretrained_vars')
    saver.restore(session, ckpt)

    print("  loaded %s model parameters from %s, time %.2fs" %
          (name, ckpt, time.time() - start_time))
    return model


def create_or_load_model(model, model_dir, session, restore_mode, name):
    """Create model and initialize or load parameters in session."""
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if restore_mode and latest_ckpt:
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
        model = load_model(model, latest_ckpt, session, restore_mode, name)
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print("# Created %s model with fresh parameters, time %.2fs" %
              (name, time.time() - start_time))

    # global_step = model.global_step.eval(session=session)
    global_step = 0
    return model, global_step
