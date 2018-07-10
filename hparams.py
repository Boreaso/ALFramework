import argparse


def add_arguments(parser: argparse.ArgumentParser):
    """Build ArgumentParser."""
    assert isinstance(parser, argparse.ArgumentParser)

    # Global params
    parser.add_argument("--param_file", type=str, default='',
                        help="Parameters file path.")
    parser.add_argument("--framework_type", type=str, default='random',
                        help="Type of framework, random | dpc | edpc | entropy | entropy_pl | egl")
    parser.add_argument("--sub_dir", type=str, default='speech_command',
                        help="Stats sub dir.")
    parser.add_argument("--num_total", type=int, default=12500,
                        help="Total num of samples to select from dataset.")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Num of classes of the classification task.")
    parser.add_argument("--sel_thresholds", type=list, default=None,
                        help="A list of thresholds for the selection of historical samples.")
    parser.add_argument("--test_percent", type=float, default=0.2,
                        help="Samples for test usage percentage of the whole dataset.")
    parser.add_argument("--valid_percent", type=float, default=0,
                        help="Samples for valid usage percentage of the whole dataset.")
    parser.add_argument("--random_seed", type=int, default=1,
                        help="Random seed.")

    # Framework params
    parser.add_argument("--labeled_percent", type=float, default=0.1,
                        help="Initial labeled samples percentage of the whole training dataset.")
    parser.add_argument("--max_round", type=int, default=60,
                        help="Max rounds of framework iteration.")
    parser.add_argument("--num_select_per_round", type=int, default=200,
                        help="Num of samples to select per iteration.")
    parser.add_argument("--pre_train", type=bool, default=False,
                        help="Whether to load the pretrained model.")
    parser.add_argument("--decay_threshold", type=float, default=0.046,
                        help="The decay threshold of entropy framework.")
    parser.add_argument("--decay_rate", type=float, default=0.001,
                        help="The decay rate of entropy framework.")
    parser.add_argument("--hist_sel_mode", type=str, default=True,
                        help="Historical samples select mode, certain | var | no")

    # Model params
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Mini batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.0005,
                        help="Learning rate of the model.")
    parser.add_argument("--metric_baseline", type=float, default=0.686824,
                        help="Model accuracy over the baseline can be saved.")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Total epochs to train.")
    parser.add_argument("--load_pretrained", type=bool, default=True,
                        help="Whether to load the pretrained model.")
