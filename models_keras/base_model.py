import abc
import logging
import math
import os
import time

import keras.backend as K
import numpy as np
from keras import Model
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
from keras.utils import to_categorical

from utils import misc_utils as utils

logger = logging.getLogger("KerasModelLogger")
logger.setLevel(level=logging.INFO)


class MyModelCheckPoint(ModelCheckpoint):
    def __init__(self, metric_baseline, **kwargs):
        super(MyModelCheckPoint, self).__init__(**kwargs)
        self.best = metric_baseline

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if self.monitor_op(current, self.best):
            logger.info('##Epoch %05d: %s improved from %0.5f to %0.5f,'
                        ' saving model to %s'
                        % (epoch + 1, self.monitor, self.best,
                           current, self.filepath))

        super(MyModelCheckPoint, self).on_epoch_end(epoch, logs)


class BaseModel:
    """Build a classifier with some convenient wraps."""

    def __init__(self,
                 class_weights,
                 input_shape,
                 num_classes=2,
                 batch_size=32,
                 embedding_size=128,
                 learning_rate=0.001,
                 metric_baseline=0.0,
                 num_epochs=20,
                 load_pretrained=False,
                 feature_type='raw',
                 output_dir=None,
                 model_dir=None):
        self._class_weights = class_weights
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._embedding_size = embedding_size
        self._learning_rate = learning_rate
        self._metric_baseline = metric_baseline
        self._num_epochs = num_epochs
        self._load_pretrained = load_pretrained
        self._feature_type = feature_type

        # The last layer's activation and loss function.
        self._last_activation = 'sigmoid' if num_classes == 1 else 'softmax'
        self._loss_func = 'binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy'

        # Specified in self._build_model() function.
        self._embedding_output_layer_name = None
        self._classifier_input_layer_name = None
        self._classifier_output_layer_name = None
        self._gradients_layer_name = None

        assert os.path.isdir(output_dir)
        self._model_dir = model_dir
        utils.ensure_path_exist(self._model_dir)
        self._log_dir = os.path.join(output_dir, 'logs')
        utils.ensure_path_exist(self._log_dir)

        log_path = os.path.join(
            self._log_dir, 'logs_' + time.ctime().replace("  ", " ")
                           .replace(" ", "_").replace(":", "_") + ".txt")

        # Logging to file.
        logger_file_handler = logging.FileHandler(log_path)
        logger.addHandler(logger_file_handler)
        # Logging to console
        logger_console_handler = logging.StreamHandler()
        logger.addHandler(logger_console_handler)

        # Build model and functions
        self._model = self._build_model()
        self._init_embedding_func()
        self._init_gradients_func()
        # self._init_total_loss_func()

        # Init model checkpoint.
        self._model_path = os.path.join(self._model_dir, '%s_model.h5' % self._model.name)
        self._model_ckpt = MyModelCheckPoint(
            metric_baseline=self._metric_baseline,
            filepath=self._model_path, monitor='val_acc', verbose=0, save_best_only=True,
            save_weights_only=False, mode='auto', period=1)

        if self._load_pretrained:
            self._load_weights()

        logger.info(
            "Model initialized: learning_rate=%f, batch_size=%d, "
            "class_weights=%s, num_epochs=%d" %
            (learning_rate, batch_size, str(class_weights), num_epochs))

    @property
    def input_shape(self):
        return [dim.value if dim.value else -1
                for dim in self._model.input.shape.dims]

    @property
    def output_shape(self):
        return [dim.value if dim.value else -1
                for dim in self._model.output.shape.dims]

    @property
    def num_classes(self):
        return self._num_classes

    @abc.abstractmethod
    def _build_model(self):
        """
        Build model architecture with keras.
        :rtype: Model
        """
        pass

    def _init_embedding_func(self):
        # make embedding function
        _input_tensors = [self._model.inputs[0]]
        _embedding_layer = self._model.get_layer(name=self._embedding_output_layer_name)
        self._embedding_func = K.function(
            inputs=_input_tensors,
            outputs=[_embedding_layer.output])

        logger.info("# Embedding function inited, embedding layer: \n %s" % _embedding_layer)

    def _init_gradients_func(self):
        # make gradients function
        _input_tensors = [self._model.inputs[0],  # inputs X
                          self._model.targets[0],  # inputs Y
                          self._model.sample_weights[0]]  # sample_weights must be specified]
        if not self._gradients_layer_name:
            _weights = self._model.trainable_weights
        else:
            _weights = self._model.get_layer(name=self._gradients_layer_name).trainable_weights
        _gradients = self._model.optimizer.get_gradients(self._model.total_loss, _weights)
        self._gradient_func = K.function(
            inputs=_input_tensors,
            outputs=_gradients)

        logger.info("# Gradients function inited, weights: ")
        for _weight in _weights:
            logger.info(" %s" % str(_weight))

    def _init_total_loss_func(self):
        # make total loss function
        _input_tensors = [self._model.inputs[0]]
        self._total_loss_func = K.function(
            _input_tensors, [self._model.total_loss])

        logger.info("# Total loss function inited, loss:\n %s" % str(self._model.total_loss))

    def _load_weights(self):
        ckpts = os.listdir(self._model_dir)
        if len(ckpts) > 0:
            model_path = os.path.join(self._model_dir, ckpts[-1])
            self._model.load_weights(filepath=model_path, by_name=True, skip_mismatch=True)
            logger.info("Weights loaded from %s" % model_path)

    def _save_weights(self, path):
        self._model.save_weights(filepath=path)
        logger.info("Weights saved to %s" % path)

    def reload_weights(self):
        self._load_weights()

    def get_model_summary(self):
        self._model.summary()

        logger.info("Trainable layers:")
        for i, layer in enumerate(self._model.layers):
            if layer.trainable:
                logger.info("layer %d: %s, %s" % (i, layer.name, str(layer)))

        logger.info("Non trainable layers:")
        for i, layer in enumerate(self._model.layers):
            if not layer.trainable:
                logger.info("layer %d: %s, %s" % (i, layer.name, str(layer)))

    def freeze_top_layers(self, end_layer_name=None):
        """Making top layers non trainable, 'end_layer_name'
        is the name of last non trainable layer."""
        if end_layer_name:
            assert end_layer_name in [layer.name for layer in self._model.layers]

            logger.info("Freezing top layers, non trainable end layer: %s" % end_layer_name)
            trainable = False
            for layer in self._model.layers:
                layer.trainable = trainable
                if layer.name == end_layer_name:
                    trainable = True

            # Recompile
            self._model.compile(
                optimizer=self._model.optimizer, loss=self._model.loss,
                metrics=self._model.metrics)

            logger.info("Current trainable layers:")
            for i, layer in enumerate(self._model.layers):
                if layer.trainable:
                    logger.info("layer %d: %s, %s" % (i, layer.name, str(layer)))

    def get_classify_func(self):
        # test mode
        K.set_learning_phase(0)

        # make function
        _input = self._model.get_layer(self._classifier_input_layer_name).input
        _output = self._model.get_layer(self._classifier_output_layer_name).output
        _classify_func = K.function(inputs=[_input], outputs=[_output])

        return _classify_func

    def get_entropy(self, inputs):
        """Binary (0，0.69315]"""
        # test mode
        # K.set_learning_phase(0)
        # _total_loss = self._total_loss_func([inputs])[0]
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)

        _preds = self.predict(inputs)

        def _compute_entropy(probs):
            assert isinstance(probs, np.ndarray)
            eps = np.spacing(1)
            if probs.shape[1] == 1:
                _entropy = -(probs * np.log(probs + eps) + (1 - probs) * np.log(1 - probs + eps))
            else:
                _entropy = np.sum(-(probs * np.log(probs + eps)), axis=1)
            return _entropy.reshape(-1)

        return _preds, _compute_entropy(_preds)

    @staticmethod
    def _make_batches(num_samples, batch_size):
        num_batches = int(math.ceil(num_samples / batch_size))
        return [(i * batch_size, min(num_samples, (i + 1) * batch_size))
                for i in range(num_batches)]

    def get_gradients(self, inputs):
        """
        Compute specified layer's gradients for unlabeled samples'
        with predict labels.
        :param inputs:
        :return: A tuple (predictions, gradients)
        """
        # Test mode
        K.set_learning_phase(0)

        # Get predict labels.
        print('Predicting...')
        _preds = self._model.predict(inputs)
        _pred_labels = np.argmax(_preds, axis=1).reshape(len(inputs), 1)

        # Compute gradients with the predicted label for each sample.
        # _batches = self._make_batches(np.alen(inputs), self._batch_size)
        _gradients = []
        for (_input, _label) in tqdm(zip(inputs, _pred_labels), desc='Computing gradients'):
            _batch_gradients = self._gradient_func([[_input], [_label], [1]])[0]
            _gradients.append(_batch_gradients)

        _gradients = np.array(_gradients)
        logger.info("Computing gradients finished, total num: %d" % np.alen(_gradients))

        return _preds, np.array(_gradients)

    def get_egl(self, inputs):
        """
        Compute expected gradient length
        (https://pdfs.semanticscholar.org/487e/d99e00bf6803a53a6059ceccd1510a63e72d.pdf)
        :param inputs:
        :return: a ndarray
        """
        # Test mode
        K.set_learning_phase(0)

        # Get predict labels.
        print('Predicting...')
        _preds = self._model.predict(inputs)
        # _pred_labels = np.argmax(_preds, axis=1).reshape(len(inputs), 1)

        # Compute gradients with the predicted label for each sample.
        # _batches = self._make_batches(np.alen(inputs), 1)
        _egls = []
        for i in tqdm(range(len(inputs)), desc='Computing egl...'):
            _egl = 0
            # compute gradients length of inputs[i]
            for j in range(len(_preds[0])):
                _label = to_categorical(j, self.num_classes)  # (m, num_classes)
                _gradients = self._gradient_func([[inputs[i]], [_label], [1]])[0]  # (m, num_gradients)
                _norm = np.linalg.norm(_gradients)  # float
                _egl += _preds[i][j] * _norm
            _egls.append(_egl)

        _egls = np.array(_egls)
        logger.info("Computing gradients finished, total num: %d" % np.alen(_egls))

        return _egls

    def embedding(self, inputs):
        """Get embedding layer's outputs."""
        # test mode
        K.set_learning_phase(0)

        # feed data
        _batches = self._make_batches(np.alen(inputs), self._batch_size)

        _embeddings = []
        for (start, end) in tqdm(_batches, desc='Embedding'):
            _batch_embeddings = self._embedding_func([inputs[start:end]])[0]
            _embeddings.extend(_batch_embeddings)

        logger.info("Embedding finished, total num: %d" % np.alen(_embeddings))

        return _embeddings

    def train(self, train_inputs, train_lablels,
              val_inputs=None, val_labels=None,
              validation_split=0.2):
        assert os.path.exists(self._model_dir) and os.path.isdir(self._model_dir)

        # train mode
        K.set_learning_phase(1)

        start_time = time.time()

        if val_inputs is not None and val_labels is not None:
            history = self._model.fit(
                x=train_inputs, y=train_lablels, batch_size=self._batch_size,
                epochs=self._num_epochs, class_weight=self._class_weights,
                validation_data=(val_inputs, val_labels), callbacks=[self._model_ckpt])
        else:
            history = self._model.fit(
                x=train_inputs, y=train_lablels, batch_size=self._batch_size,
                epochs=self._num_epochs, class_weight=self._class_weights,
                validation_split=validation_split, callbacks=[self._model_ckpt])

        logger.info(history.history)
        logger.info('# Training finished, total time: %.4f.' % (time.time() - start_time))

    def evaluate(self, eval_inputs, eval_labels):
        # test mode
        K.set_learning_phase(0)

        result = self._model.evaluate(
            x=eval_inputs, y=eval_labels,
            batch_size=self._batch_size,
            verbose=0)

        logger.info('# Evaluation finished, result: %s' % result)

        return result

    def predict(self, pred_inputs):
        # test mode
        K.set_learning_phase(0)

        return self._model.predict(
            x=pred_inputs, verbose=0)
