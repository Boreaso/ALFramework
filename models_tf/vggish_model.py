import tensorflow as tf
from tensorflow.contrib import slim

from . import model_helper as helper
from .base_model import BaseModel


class VggishModel(BaseModel):
    """Defines the VGGish TensorFlow model.

    All ops are created in the current default graph, under the scope 'vggish/'.
    The input is a placeholder named 'vggish/input_features' of type float32 and
    shape [batch_size, num_frames, num_bands] where batch_size is variable and
    num_frames and num_bands are constants, and [num_frames, num_bands] represents
    a log-mel-scale spectrogram patch covering num_bands frequency bands and
    num_frames time frames (where each frame step is usually 10ms). This is
    produced by computing the stabilized log(mel-spectrogram + params.LOG_OFFSET).
    The output is an op named 'vggish/embedding' which produces the activations of
    a 128-D embedding layer, which is usually the penultimate layer when used as
    part of a full model with a final classifier layer.
    """

    def _build_embedding_layers(self):
        # Trainable check.
        trainable = self._check_trainable()
        batched_features = self._batched_features
        embedding_size = self._embedding_size
        stddev = self._init_stddev
        dropout = self._dropout

        # Defaults:
        # - All weights are initialized to N(0, INIT_STDDEV).
        # - All biases are initialized to 0.
        # - All activations are ReLU.
        # - All convolutions are 3x3 with stride 1 and SAME padding.
        # - All max-pools are 2x2 with stride 2 and SAME padding.
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=helper.get_initializer('truncated_normal', stddev=stddev),
                            biases_initializer=helper.get_initializer('zeros'),
                            activation_fn=tf.nn.relu,
                            trainable=trainable), \
             slim.arg_scope([slim.conv2d], kernel_size=[3, 3], stride=1, padding='SAME'), \
             slim.arg_scope([slim.max_pool2d], kernel_size=[2, 2], stride=2, padding='SAME'), \
             tf.variable_scope('vggish'):
            # # Input: a batch of 2-D log-mel-spectrogram patches.
            # features = tf.placeholder(
            #     tf.float32, shape=(None, params.NUM_FRAMES, params.NUM_BANDS),
            #     name='input_features')
            # # Reshape to 4-D so that we can convolve a batch with conv2d().
            # net = tf.reshape(features, [-1, params.NUM_FRAMES, params.NUM_BANDS, 1])
            batched_features = tf.reshape(batched_features, [-1, 96, 64, 1])
            # The VGG stack of alternating convolutions and max-pools.
            net = slim.conv2d(batched_features, 64, scope='conv1')
            net = slim.max_pool2d(net, scope='pool1')
            net = slim.conv2d(net, 128, scope='conv2')
            net = slim.max_pool2d(net, scope='pool2')
            net = slim.repeat(net, 2, slim.conv2d, 256, scope='conv3')
            net = slim.max_pool2d(net, scope='pool3')
            net = slim.repeat(net, 2, slim.conv2d, 512, scope='conv4')
            net = slim.max_pool2d(net, scope='pool4')

            # Flatten before entering fully-connected layers
            net = slim.flatten(net)
            net = slim.repeat(net, 2, slim.fully_connected, 4096, scope='fc1')
            # The embedding layer.
            net = slim.fully_connected(net, embedding_size, scope='fc2')
            return tf.identity(net, name='embedding')

    def _build_classifier(self, embeddings=None):
        batched_labels = self._batched_labels
        batch_size = tf.size(batched_labels)
        num_units = self._num_units
        num_classes = self._num_classes

        # Define a shallow classification model and associated training ops on top
        # of VGGish.
        with tf.variable_scope('classifier'):
            # Add a fully connected layer with 100 units.

            fc = embeddings
            for num_unit in num_units:
                fc = slim.fully_connected(fc, num_unit)

            # Add a classifier layer at the end, consisting of parallel logistic
            # classifiers, one per class. This allows for multi-class tasks.
            logits = slim.fully_connected(
                fc, num_classes, activation_fn=None, scope='logits')
            probabilities = tf.nn.softmax(logits, name='softmax_logits')

            # Cross entropy and loss
            onehot_labels = tf.one_hot(
                indices=batched_labels, depth=self._num_classes)
            cross_ent = tf.nn.weighted_cross_entropy_with_logits(
                logits=logits, targets=onehot_labels,
                pos_weight=self._class_weights, name='loss_op')

            loss = tf.reduce_sum(cross_ent) / tf.to_float(batch_size)

            return {'logits': logits,
                    'probabilities': probabilities,
                    'entropy': cross_ent,
                    'loss': loss}
