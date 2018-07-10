import keras
import numpy as np
from keras.layers import BatchNormalization, Conv2D, Dense, \
    Dropout, Flatten, InputLayer, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical

from models_keras.base_model import BaseModel
from models_keras.keras_metrics import f1score, precision, recall
from utils import data_utils


class VggishModel(BaseModel):
    """Build a vgg-16 like model with keras."""

    def _build_model(self):
        model = keras.Sequential(name='vggish')

        if self._feature_type == 'raw':
            # Input
            model.add(InputLayer(input_shape=self._input_shape, name='input_layer'))
            model.add(BatchNormalization(name='input_batch_norm'))

            # Block 1
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))  # 96 * 64 * 64
            model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))  # 48 * 32 * 64

            # Block 2
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))  # 48 * 32 * 128
            model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))  # 24* 16 * 128
            model.add(BatchNormalization(name='block2_batch_norm'))
            model.add(Dropout(0.5, name='block2_dropout'))

            # Block 3
            # model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))  # 24 * 16 * 256
            # model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))  # 24 * 16 * 256
            # model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))  # 12 * 8 * 256
            # model.add(BatchNormalization(name='block3_batch_norm'))
            # model.add(Dropout(0.5, name='block3_dropout'))

            # Block 4
            # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))  # 12 * 8 * 512
            # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))  # 12 * 8 * 512
            # model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))  # 6 * 4 * 512
            # model.add(BatchNormalization(name='block4_batch_norm'))
            # model.add(Dropout(0.5, name='block4_dropout'))

            # Embedding block
            model.add(Flatten(name='flatten'))  # 12288
            # model.add(Dense(128, activation='relu', name='embedding_1'))  # 4096
            # model.add(BatchNormalization(name='embedding_batch_norm_1'))
            # model.add(Dropout(0.5, name='embedding_dropout_1'))
            # model.add(Dense(4096, activation='relu', name='embedding_2'))
            # model.add(BatchNormalization(name='embedding_batch_norm_2'))
            # model.add(Dropout(0.5, name='embedding_dropout_2'))
            model.add(Dense(self._embedding_size, activation='relu', name='embedding_3'))  # 128
        elif self._feature_type == 'embedding':
            model.add(InputLayer(input_shape=(self._embedding_size,), name='input_layer'))
        else:
            raise ValueError("Unknown feature type: %s." % self._feature_type)

        # Classification block
        model.add(BatchNormalization(name='embedding_batch_norm_1'))
        model.add(Dropout(0.5, name='embedding_dropout_1'))
        model.add(Dense(32, activation='relu', name='classification_1', kernel_regularizer='l2'))  # 64
        # model.add(Dense(32, activation='relu', name='classification_2'))  # 32
        model.add(Dense(self._num_classes, activation=self._last_activation,
                        name='prediction', kernel_regularizer='l2'))  # 1

        optimizer = Adam(lr=self._learning_rate)
        model.compile(
            optimizer=optimizer, loss=self._loss_func,
            metrics=['accuracy', precision, recall, f1score])

        # Init special field.
        self._embedding_output_layer_name = 'embedding_3'
        self._classifier_input_layer_name = 'classification_1'
        self._classifier_output_layer_name = 'prediction'
        self._gradients_layer_name = 'prediction'

        return model


if __name__ == '__main__':
    _num_total = 10000

    _pairs = data_utils.load_data('../data/urbansound8k/pairs')
    _pairs = np.random.choice(_pairs, _num_total, replace=False)

    # Extract features and labels.
    _train_pairs, _, test_pairs = data_utils.split_dataset(
        pairs=_pairs, valid_percent=0, test_percent=0.2, num_classes=10)

    _input_shape = [60, 41, 2]
    _labels = np.array(_pairs['label'].tolist())
    _class_weights = [_num_total / sum(_labels == i) for i in range(10)]

    _train_features = np.reshape(_train_pairs['feature'].tolist(), [-1] + _input_shape)
    _train_labels = to_categorical(_train_pairs['label'].tolist())
    _test_features = np.reshape(test_pairs['feature'].tolist(), [-1] + _input_shape)
    _test_labels = to_categorical(test_pairs['label'].tolist())

    _vggish_model = VggishModel(
        class_weights=_class_weights, input_shape=_input_shape,
        num_classes=10, batch_size=64, learning_rate=0.0005,
        metric_baseline=0.94, num_epochs=50, load_pretrained=False,
        feature_type='raw', output_dir='outputs')

    _vggish_model.get_model_summary()

    # vggish_model.freeze_top_layers('block4_pool')
    _vggish_model.train(_train_features, _train_labels,
                        _test_features, _test_labels)
    # vggish_model.evaluate(_test_features, _test_labels)
    # embeddings = vggish_model.embedding()
    # data_utils.save_data(np.array(embeddings), file_path='../data/vggish_embeddings')

    # entropy = vggish_model.get_entropy(_features[:10])
    # print(entropy)

    # gradients = vggish_model.get_gradients(_features[:10])
    # print(gradients)
