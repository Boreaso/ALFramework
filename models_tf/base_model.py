import abc

import tensorflow as tf
from tensorflow.estimator import ModeKeys

from models_tf.dataset import DatasetIterator


class BaseModel:

    def __init__(self,
                 iterator,
                 learning_rate,
                 batch_size,
                 num_classes,
                 embedding_size=128,
                 num_units=100,
                 dropout=0.5,
                 class_weights=None,
                 init_range=None,
                 init_stddev=1.0,
                 max_gradients_norm=5,
                 mode=ModeKeys.TRAIN,
                 optimizer='sgd',
                 ckpt_dir=None,
                 num_keep_ckpts=5,
                 random_seed=123):
        # Hyper parameter
        assert isinstance(iterator, DatasetIterator)
        self._iterator = iterator
        self._batched_features = iterator.batched_features
        self._batched_targets = iterator.batched_targets

        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._num_classes = num_classes
        self._embedding_size = embedding_size
        self._num_units = num_units
        self._dropout = dropout
        self._class_weights = tf.constant(class_weights)
        self._init_range = init_range
        self._init_stddev = init_stddev
        self._max_gradients_norm = max_gradients_norm
        self._mode = mode
        self._optimizer = optimizer
        self._ckpt_dir = ckpt_dir
        self._num_keep_ckpts = num_keep_ckpts
        self._random_seed = random_seed

        # Build embedding layers
        self._embeddings = self._build_embedding_layers()

        # Store all vars in embedding layers
        self._embedding_vars = [var for var in tf.global_variables()]

        # Global step
        self._global_step = tf.Variable(0, trainable=False, name='global_step')

        # Add classifier layers
        result = self._build_classifier(self._embeddings)

        # Current trainable variables
        self._variables = tf.trainable_variables()

        self._logits = result['logits']
        self._train_loss = result['loss']

        if self._mode == ModeKeys.TRAIN:
            # Optimizer
            if self._optimizer == "sgd":
                opt = tf.train.GradientDescentOptimizer(self._learning_rate)
            elif self._optimizer == 'rmsprop':
                opt = tf.train.RMSPropOptimizer(self._learning_rate)
            elif self._optimizer == 'adam':
                opt = tf.train.AdamOptimizer(self._learning_rate)
            else:
                raise ValueError('Invalid `optimizer` value.')

            # Gradients
            self._gradients = tf.gradients(self._train_loss, self._variables)
            self._clipped_gradients, self._gradients_norm = tf.clip_by_global_norm(
                self._gradients, self._max_gradients_norm)

            # Update op
            self._update = opt.apply_gradients(
                grads_and_vars=zip(self._gradients, self._variables),
                global_step=self._global_step)

            # # Estimator
            # self._estimator = self._build_estimator()

            # Summary
            self._train_summary = tf.summary.merge(
                [tf.summary.scalar("lr", self._learning_rate),
                 tf.summary.scalar("train_loss", self._train_loss),
                 tf.summary.scalar("grad_norm", self._gradients_norm),
                 tf.summary.scalar("clipped_gradient", self._gradients_norm)])

        # Print trainable variables
        print("# Trainable variables")
        for var in self._variables:
            print("  %s, %s, %s" % (var.name, str(var.get_shape()),
                                    var.op.device))

        # Store all vars in classifier layers
        self._classifier_vars = [var for var in tf.global_variables() if var not in self._embedding_vars]

        # Saver
        self._saver = tf.train.Saver(
            tf.global_variables(), allow_empty=True,
            max_to_keep=self._num_keep_ckpts)

    def _check_trainable(self):
        return self._mode == ModeKeys.TRAIN

    def _get_estimator_mode(self):
        return self._mode

    @abc.abstractmethod
    def _build_embedding_layers(self):
        """Build some first layers of the model.
        :rtype: object
        """
        pass

    @abc.abstractmethod
    def _build_classifier(self, _embeddings=None):
        """Build classifier.
        :return dict, keys: 'loss', 'logits'
        """
        pass

    # def _build_estimator(self):
    #     predictions = tf.argmax(self._logits)
    #     eval_metric_ops = {"Loss": self._train_loss,
    #                        "Accuracy": tf.metrics.accuracy(
    #                            labels=self._batched_targets,
    #                            predictions=predictions,
    #                            name='accuracy')}
    #
    #     model_fn = tf.estimator.EstimatorSpec(
    #         mode=self._get_estimator_mode(),
    #         predictions=predictions,
    #         loss=self._train_loss,
    #         train_op=self._update,
    #         eval_metric_ops=eval_metric_ops)
    #
    #     estimator = tf.estimator.Estimator(
    #         model_fn=model_fn, model_dir=self._ckpt_dir)
    #
    #     return estimator

    def embedding(self, sess):
        """Extract features with pretrained weights."""
        assert self._mode == ModeKeys.EMBEDDING
        return sess.run(self._embeddings)

    def train(self, sess):
        assert self._mode == ModeKeys.TRAIN
        return sess.run([
            self._global_step,
            self._train_loss,
            self._train_summary])

    def predict(self, sess):
        assert self._mode == ModeKeys.PREDICT
        prediction = tf.argmax(self._logits)
        return sess.run(prediction)

    def load_weights(self, sess):
        assert isinstance(sess, tf.Session)
        if self._ckpt_dir:
            # Get the list of names of all VGGish variables that exist in
            # the checkpoint (i.e., all inference-mode VGGish variables).
            with sess.graph.as_default():
                model_var_names = [v.name for v in tf.global_variables()]

            # Get the list of all currently existing variables that match
            # the list of variable names we just computed.
            model_vars = [v for v in tf.global_variables() if v.name in model_var_names]

            # Use a Saver to restore just the variables selected above.
            saver = tf.train.Saver(model_vars, name='model_load_pretrained')
            saver.restore(sess, self._ckpt_dir)
        else:
            raise ValueError("Invalid param 'ckpt_path'")
