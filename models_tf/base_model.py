import abc

import tensorflow as tf


class ModeKeys(object):
    """Standard names for model modes.
    The following standard keys are defined:
    * `TRAIN`: training mode.
    * `TRAIN_CLASSIFIER`: train model without embedding layers.
    * `FINE_TUNE`: fine tuning model.
    * `EMBEDDING`: get embeddings without classifier.
    * `PREDICT`: prediction mode.
    * `EVAL`: evaluation mode.
    """
    TRAIN = 'train'
    TRAIN_CLASSIFIER = 'train_classifier'
    FINE_TUNE = 'fine_tune'
    PREDICT = 'predict'
    EMBEDDING = 'embedding'
    EVAL = 'eval'


class BaseModel:

    def __init__(self,
                 iterator,
                 learning_rate,
                 num_classes,
                 embedding_size=128,
                 num_units=100,
                 dropout=0.5,
                 class_weights=None,
                 init_range=None,
                 init_stddev=1.0,
                 mode=ModeKeys.TRAIN,
                 optimizer='sgd',
                 ckpt_dir=None,
                 num_keep_ckpts=5,
                 random_seed=123):
        # Data inputs
        self._batched_features = iterator.batched_features
        self._batched_labels = iterator.batched_labels

        # Hyper parameters
        self._learning_rate = tf.constant(learning_rate)
        self._num_classes = num_classes
        self._embedding_size = embedding_size
        self._num_units = num_units
        self._dropout = dropout
        self._class_weights = tf.constant(class_weights)
        self._init_range = init_range
        self._init_stddev = init_stddev
        self._mode = mode
        self._optimizer = optimizer
        self._ckpt_dir = ckpt_dir
        self._num_keep_ckpts = num_keep_ckpts
        self._random_seed = random_seed

        # Build embedding layers
        self._embeddings = self._build_embedding_layers() \
            if self._mode != ModeKeys.TRAIN_CLASSIFIER else self._batched_features
        # Store all vars in embedding layers
        self.embedding_vars = [var for var in tf.global_variables()]

        # Global step
        self._global_step = tf.Variable(0, trainable=False, name='global_step')

        if self._mode != ModeKeys.EMBEDDING:
            # Add classifier layers
            result = self._build_classifier(self._embeddings)

            self._predictions = tf.argmax(self._logits, axis=-1)
            self.eval_metric_ops = {"Accuracy": tf.metrics.accuracy(
                labels=self._batched_labels, predictions=self._predictions, name='accuracy')}

            # Current trainable variables
            self._variables = tf.trainable_variables()

            if self._mode == ModeKeys.TRAIN or self._mode == ModeKeys.TRAIN_CLASSIFIER \
                    or self._mode == ModeKeys.FINE_TUNE:
                self._train_loss = result['loss']
                self._logits = result['logits']

                # Optimizer
                if self._optimizer == "sgd":
                    opt = tf.train.GradientDescentOptimizer(self._learning_rate)
                    tf.summary.scalar("lr", self._learning_rate)
                elif self._optimizer == 'rmsprop':
                    opt = tf.train.RMSPropOptimizer(self._learning_rate)
                else:
                    opt = tf.train.AdamOptimizer(self._learning_rate)

                # Gradients
                self._gradients = tf.gradients(self._train_loss, self._variables)
                # clipped_grads, grad_norm_summary, grad_norm = model_helper.gradient_clip(
                #     gradients, max_gradient_norm=self.max_gradient_norm)
                # self.grad_norm = grad_norm
                self._update = opt.apply_gradients(
                    grads_and_vars=zip(self._gradients, self._variables),
                    global_step=self._global_step)

                # self.estimator = self._build_estimator()

                # Summary
                self._train_summary = tf.summary.merge(
                    [tf.summary.scalar("lr", self._learning_rate),
                     tf.summary.scalar("train_loss", self._train_loss)])
            elif self._mode == ModeKeys.PREDICT:
                self._logits = result['logits']

            # Print trainable variables
            print("# Trainable variables")
            for var in self._variables:
                print("  %s, %s, %s" % (var.name, str(var.get_shape()),
                                        var.op.device))

        # Store all vars in classifier layers
        self.classifier_vars = [var for var in tf.global_variables() if var not in self.embedding_vars]

        # Saver
        self.saver = tf.train.Saver(
            tf.global_variables(), allow_empty=True,
            max_to_keep=self._num_keep_ckpts)

    def _check_trainable(self):
        return self._mode == ModeKeys.TRAIN

    def _get_estimator_mode(self):
        if self._mode == ModeKeys.TRAIN or self._mode == ModeKeys.TRAIN_CLASSIFIER \
                or self._mode == ModeKeys.FINE_TUNE:
            return 'train'
        elif self._mode == ModeKeys.PREDICT:
            return 'infer'
        elif self._mode == ModeKeys.EVAL:
            return 'eval'

    @abc.abstractmethod
    def _build_embedding_layers(self):
        """Build some first layers of the model.
        :rtype: object
        """
        pass

    @abc.abstractmethod
    def _build_classifier(self, _embeddings=None):
        """Build classifier.
        :rtype: dict
        """
        pass

    # def _build_estimator(self):
    #     predictions = tf.argmax(self._logits)
    #     eval_metric_ops = {"Accuracy": tf.metrics.accuracy(
    #         labels=self._batched_labels, predictions=predictions, name='accuracy')}
    #
    #     model_fn = tf.estimator.EstimatorSpec(
    #         mode=self._get_estimator_mode(),
    #         predictions=predictions,
    #         loss=self._train_loss,
    #         train_op=self._update,
    #         eval_metric_ops=eval_metric_ops, )
    #
    #     estimator = tf.estimator.Estimator(
    #         model_fn=model_fn, model_dir=self._ckpt_dir)
    #     return estimator

    def _embedding(self, sess):
        """Extract features with pretrained weights."""
        assert self._mode == ModeKeys.EMBEDDING
        return sess.run(self._embeddings)

    def _train(self, sess):
        assert self._mode == ModeKeys.TRAIN or \
               self._mode == ModeKeys.FINE_TUNE or \
               self._mode == ModeKeys.TRAIN_CLASSIFIER
        return sess.run(
            [self._update,
             self._gradients,
             self._train_loss,
             self.eval_metric_ops,
             self._train_summary,
             self._global_step,
             self._learning_rate])

    def _predict(self, sess):
        assert self._mode == ModeKeys.PREDICT
        return sess.run(self._logits)

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

    def run_step(self, sess):
        if self._mode == ModeKeys.TRAIN or \
                self._mode == ModeKeys.FINE_TUNE or \
                self._mode == ModeKeys.TRAIN_CLASSIFIER:
            return self._train(sess)
        elif self._mode == ModeKeys.PREDICT:
            return self._predict(sess)
        elif self._mode == ModeKeys.EMBEDDING:
            return self._embedding(sess)
        else:
            raise ValueError('Unknown mode value.')
