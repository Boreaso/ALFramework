import argparse
import os

import tensorflow as tf

from models.vggish_model import VggishModel
from utils import data_utils, param_utils
from utils import misc_utils as utils


class VggishModelEst(VggishModel):

    def _build_classifier(self, embeddings=None):
        feature_columns = tf.feature_column.numeric_column(
            'features', shape=self._embedding_size)

        classifier = tf.estimator.DNNClassifier(
            hidden_units=self._num_units,
            feature_columns=feature_columns,
            model_dir=self._ckpt_dir,
            n_classes=self._num_classes,
            weight_column='class_weights',
            optimizer='Adam',
            dropout=self._dropout)

        return classifier


def run_train(features, labels, hparams):
    out_dir = hparams.out_dir
    utils.ensure_path_exist(out_dir)
    model_dir = os.path.join(out_dir, "ckpts")
    utils.ensure_path_exist(model_dir)

    model_params = param_utils.get_model_params(hparams, None)
    model = VggishModelEst(**model_params.values())

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"features": features, "class_weights": hparams.class_weights},
        y=labels, batch_size=hparams.batch_size,
        num_epochs=hparams.num_epochs, shuffle=True)

    model.classifier.train(input_fn=input_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    param_utils.add_arguments(parser)
    FLAGS, unused = parser.parse_known_args()

    loaded_hparams = param_utils.create_hparams(FLAGS)
    json_str = open('../params/vggish_hparams.json').read()
    loaded_hparams.parse_json(json_str)

    _hparams = param_utils.create_hparams(FLAGS)
    param_utils.combine_hparams(_hparams, loaded_hparams)

    print(loaded_hparams.values())

    _features = data_utils.load_data('../data/features')
    _labels = data_utils.load_data('../data/labels')
    run_train(_features, _labels, loaded_hparams)
