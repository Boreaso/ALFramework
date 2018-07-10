import argparse
import os
import time

import numpy as np
import tensorflow as tf

from models import model_helper
from models.vggish_model import VggishModel
from utils import data_utils
from utils import misc_utils as utils
from utils import param_utils


def run_train(features, labels, hparams):
    model = hparams.model
    restore_mode = hparams.restore_mode
    summary_name = "train_summary"
    num_train_steps = hparams.num_train_steps
    steps_per_stats = hparams.steps_per_stats
    out_dir = hparams.out_dir
    utils.ensure_path_exist(out_dir)
    model_dir = os.path.join(out_dir, "ckpts")
    utils.ensure_path_exist(model_dir)

    if model == 'vggish':
        model_creator = VggishModel
    else:
        raise ValueError("Unknown model type.")

    train_model = model_helper.create_train_model(
        hparams=hparams,
        model_creator=model_creator)

    # Session
    config_proto = utils.get_config_proto()
    train_sess = tf.Session(config=config_proto, graph=train_model.graph)

    # Load train model
    with train_model.graph.as_default():
        loaded_train_model, global_step = model_helper.create_or_load_model(
            train_model.model, model_dir, train_sess, restore_mode, "train")

    # Summary writer
    summary_writer = tf.summary.FileWriter(
        os.path.join(out_dir, summary_name), train_model.graph)

    # Initialize dataset iterator
    train_sess.run(
        train_model.iterator.initializer,
        feed_dict={train_model.skip_count_placeholder: 0,
                   train_model.features_placeholder: features,
                   train_model.labels_placeholder: labels})

    training_start_time = time.time()
    epoch_count = 0
    last_stats_step = global_step
    while global_step < num_train_steps:
        # Run a training step
        try:
            _, _, train_loss, eval_res, train_summary, global_step, learning_rate = \
                loaded_train_model.run_step(train_sess)
        except tf.errors.OutOfRangeError:
            # Finished going through the training dataset. Go to next epoch.
            epoch_count += 1
            print("# Finished epoch %d, step %d." %
                  (epoch_count, global_step))

            # Save model params
            loaded_train_model.saver.save(
                train_sess,
                os.path.join(model_dir, "%s_model.ckpt" % model),
                global_step=global_step)

            train_sess.run(
                train_model.iterator.initializer,
                feed_dict={train_model.skip_count_placeholder: 0,
                           train_model.features_placeholder: features,
                           train_model.labels_placeholder: labels})
            continue

        # Write step summary and accumulate statistics
        summary_writer.add_summary(train_summary)

        if global_step - last_stats_step >= steps_per_stats:
            last_stats_step = global_step

            print("  loss: %.4f  eval res: %s" % (train_loss, str(eval_res)))

    # Training done.
    loaded_train_model.saver.save(
        train_sess,
        os.path.join(model_dir, "%s_model.ckpt" % model),
        global_step=global_step)

    summary_writer.close()

    print('Training done. Total time: %.4f' % (time.time() - training_start_time))


def run_embedding(features, labels, hparams):
    model = hparams.model
    restore_mode = hparams.restore_mode
    out_dir = hparams.out_dir
    utils.ensure_path_exist(out_dir)
    model_dir = os.path.join(out_dir, "ckpts")
    utils.ensure_path_exist(model_dir)

    if model == 'vggish':
        model_creator = VggishModel
    else:
        raise ValueError("Unknown model type.")

    embedding_model = model_helper.create_embedding_model(
        hparams=hparams,
        model_creator=model_creator)

    # Session
    config_proto = utils.get_config_proto()
    embedding_sess = tf.Session(config=config_proto, graph=embedding_model.graph)

    # Load embedding model
    with embedding_model.graph.as_default():
        loaded_embedding_model, _ = model_helper.create_or_load_model(
            embedding_model.model, model_dir, embedding_sess, restore_mode, "embedding")

    # Initialize dataset iterator
    embedding_sess.run(
        embedding_model.iterator.initializer,
        feed_dict={embedding_model.features_placeholder: features,
                   embedding_model.labels_placeholder: labels})

    # Embedding op
    print("# Start embedding.")
    embedding_start_time = time.time()
    embeddings = []
    count = 0
    while True:
        # Run a training step
        try:
            batched_embeddings = loaded_embedding_model.run_step(embedding_sess)
            embeddings.extend(batched_embeddings)
            # count += batched_embeddings.shape[0]
            # print(count)
        except tf.errors.OutOfRangeError:
            # Finished going through the dataset. Save embeddings.
            print("# Finished embedding, time %.2f." %
                  (time.time() - embedding_start_time))
            break

    print("Embedding finished, num: %d" % len(embeddings))

    data_utils.save_data(np.array(embeddings), file_path=os.path.join(out_dir, 'embeddings'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    param_utils.add_arguments(parser)
    FLAGS, unused = parser.parse_known_args()

    loaded_hparams = param_utils.create_hparams(FLAGS)
    json_str = open('params/vggish_hparams.json').read()
    loaded_hparams.parse_json(json_str)

    _hparams = param_utils.create_hparams(FLAGS)
    param_utils.combine_hparams(_hparams, loaded_hparams)

    print(loaded_hparams.values())

    _features = data_utils.load_data('data/features')
    _labels = data_utils.load_data('data/labels')
    run_embedding(_features, _labels, loaded_hparams)
