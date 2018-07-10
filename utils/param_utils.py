import argparse
import os

import tensorflow.contrib.training as tf_training


def add_arguments(parser):
    """Build ArgumentParser."""
    assert isinstance(parser, argparse.ArgumentParser)

    parser.register("type", "bool", lambda v: v.lower() == "true")

    # Data
    parser.add_argument("--mode", type=str, default='train',
                        help="""Specify the mode of program.Options include:
                        train: Training opt.
                        fine_tune: Fine tuning model with pretrained weights(freeze first embedding layers).
                        predict: Select a sentence randomly from train source file and decode.
                        embedding: Only build first embedding layers and get embeddings computed
                                   with pretrained weights.""")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory, e.g., en.")
    parser.add_argument("--train_prefix", type=str, default=None,
                        help="Train file prefix.")
    parser.add_argument("--dev_prefix", type=str, default=None,
                        help="Develop file prefix.")
    parser.add_argument("--test_prefix", type=str, default=None,
                        help="Test file prefix.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Misc data output directory")

    # Network
    parser.add_argument("--embedding_size", type=int, default=128,
                        help="Embedding layers output size(feature vector length)")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="The number of class.")
    parser.add_argument("--num_units", type=list, default=[64, 32],
                        help="The number of classifier layers' units(network size).")
    parser.add_argument("--class_weights", type=list, default=[1.0, 1.0],
                        help="Class weights to trade-off class imbalance.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate (not keep_prob)")
    parser.add_argument("--init_range", type=float, default=0.1,
                        help="""Weight range for uniform initializer.""")
    parser.add_argument("--init_stddev", type=float, default=1.0,
                        help="Weight stddev for truncated_normal initializer.")

    # Train
    parser.add_argument("--model", type=str, default="vggish",
                        help="vggish")
    parser.add_argument("--restore_mode", type=str, default=None,
                        help="""
                        None: don't restore vars.
                        all: restore all vars in model checkpoint.
                        embedding: restore only vars in embedding layers.
                        classifier: restore only vars in classifier layers.""")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference mode.")
    parser.add_argument("--optimizer", type=str, default="sgd",
                        help="sgd | rmsprop | adam")
    parser.add_argument("--learning_rate", type=float, default=1.0,
                        help="Learning rate. Adam: 0.001 | 0.0001")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Num epochs to train.")
    parser.add_argument("--num_train_steps", type=int, default=10000,
                        help="Num steps to train.")
    parser.add_argument("--steps_per_stats", type=int, default=100,
                        help="Num steps to log stats.")
    parser.add_argument("--num_keep_ckpts", type=int, default=5,
                        help="Max checkpoints to be saved.")
    parser.add_argument("--shuffle", type=bool, default=True,
                        help="Whether to apply shuffle ops on dataset.")
    parser.add_argument("--random_seed", type=int, default=123,
                        help="Random seed.")


def create_hparams(flags):
    """Create training hparams."""
    return tf_training.HParams(
        # mode
        mode=flags.mode,
        # Data
        data_dir=flags.data_dir,
        train_prefix=flags.train_prefix,
        dev_prefix=flags.dev_prefix,
        test_prefix=flags.test_prefix,
        out_dir=flags.out_dir,
        # Networks
        embedding_size=flags.embedding_size,
        num_units=flags.num_units,
        num_classes=flags.num_classes,
        class_weights=flags.class_weights,
        dropout=flags.dropout,
        init_range=flags.init_range,
        init_stddev=flags.init_stddev,
        # Train
        model=flags.model,
        restore_mode=flags.restore_mode,
        batch_size=flags.batch_size,
        optimizer=flags.optimizer,
        learning_rate=flags.learning_rate,
        num_epochs=flags.num_epochs,
        num_train_steps=flags.num_train_steps,
        steps_per_stats=flags.steps_per_stats,
        shuffle=flags.shuffle,
        num_keep_ckpts=flags.num_keep_ckpts,
        random_seed=flags.random_seed)


def get_model_params(hparams,
                     iterator):
    ckpt_dir = os.path.join(hparams.out_dir, 'ckpts')
    return tf_training.HParams(
        iterator=iterator,
        learning_rate=hparams.learning_rate,
        num_classes=hparams.num_classes,
        embedding_size=hparams.embedding_size,
        num_units=hparams.num_units,
        dropout=hparams.dropout,
        class_weights=hparams.class_weights,
        init_range=hparams.init_range,
        init_stddev=hparams.init_stddev,
        mode=hparams.mode,
        optimizer=hparams.optimizer,
        ckpt_dir=ckpt_dir,
        num_keep_ckpts=hparams.num_keep_ckpts,
        random_seed=hparams.random_seed)


def combine_hparams(hparams, new_hparams):
    assert hparams and new_hparams

    loaded_config = hparams.values()
    for key in loaded_config:
        if getattr(new_hparams, key) != loaded_config[key]:
            print("# Updating hparams.%s: %s -> %s" %
                  (key, str(loaded_config[key]),
                   str(getattr(new_hparams, key))))
            setattr(hparams, key, getattr(new_hparams, key))
    return hparams


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    add_arguments(_parser)
    _FLAGS, _unused = _parser.parse_known_args()

    _loaded_hparams = create_hparams(_FLAGS)
    _json_str = open('../hparams/chatbot_xhj.json').read()
    _loaded_hparams.parse_json(_json_str)

    _hparams = create_hparams(_FLAGS)

    combine_hparams(_hparams, _loaded_hparams)

    print(_loaded_hparams.values())
