import logging
import gin
import tensorflow as tf
from models.architectures import (simple_rnn, simple_lstm, simple_gru, bidirectional_rnn,
                                  bidirectional_lstm, bidirectional_gru,
                                  stacked_rnn, stacked_lstm, stacked_gru)
from train import Trainer
from utils import utils_params, utils_misc

from tensorboard.plugins.hparams import api as hp
from evaluation.evaluation import Evaluator
from input_pipeline import datasets


@gin.configurable
def hyperparameter_tuning(models, run_paths,
                          simple_models_params, stacked_two_layer_models_params):  # <- configs
    """Train and evaluate the given models with varying hyperparameter."""

    simple_models = ['simple_rnn', 'simple_lstm', 'simple_gru',
                     'bidirectional_rnn', 'bidirectional_lstm', 'bidirectional_gru']

    stacked_models = ['stacked_rnn', 'stacked_lstm', 'stacked_gru']

    for model in models:
        model_name = model['model']  # this is the model dict defined in config

        # Make new folder inside 'hyperparameter_tuning' folder based on model names
        utils_params.generate_hyperparameter_model_directories(model_name, run_paths)

        # Setup summary writer
        hyperparameter_logs_directory = run_paths['hyperparameter_tuning_directory'] / model_name
        hyperparameter_summary_writer = tf.summary.create_file_writer(logdir=str(hyperparameter_logs_directory))

        # Metric for hyperparameter comparison that is common among all the models
        HP_METRIC_LOSS = 'loss'
        HP_METRIC_ACCURACY = 'accuracy'

        logging.info(f"Starting hyper-parameter tuning for model {model_name} ->")

        if model_name in simple_models:
            HP_DATASET_WINDOW_SIZE = hp.HParam('dataset_window_size',
                                               hp.Discrete(simple_models_params['dataset_window_size']))
            HP_DATASET_WINDOW_SHIFT_RATIO = hp.HParam('dataset_window_shift',
                                                      hp.Discrete(simple_models_params['dataset_window_shift_ratio']))
            HP_UNITS_1_RATE = hp.HParam(name='units',
                                        domain=hp.IntInterval(
                                            min_value=simple_models_params['units']['min'],
                                            max_value=simple_models_params['units']['max']))

            HP_ACTIVATION_UNITS = hp.HParam('activation',
                                            hp.Discrete(simple_models_params['activation']))

            # Generate random searches in the HP_UNITS_RATE.domain
            units_1_list = []
            units_n_search = simple_models_params['units']['n_search']
            count = 0
            while count < units_n_search:
                units_uniform = HP_UNITS_1_RATE.domain.sample_uniform()
                units_uniform = int(units_uniform)
                if units_uniform not in units_1_list:
                    units_1_list.append(units_uniform)
                    count += 1

            # Save Tensorboard hyperparameter configuration
            with hyperparameter_summary_writer.as_default():
                hp.hparams_config(
                    hparams=[HP_UNITS_1_RATE, HP_ACTIVATION_UNITS,
                             HP_DATASET_WINDOW_SIZE, HP_DATASET_WINDOW_SHIFT_RATIO],
                    metrics=[hp.Metric(tag=HP_METRIC_LOSS, display_name='Loss'),
                             hp.Metric(tag=HP_METRIC_ACCURACY, display_name='Accuracy')]
                )

            session_num = 0

            for units_1 in units_1_list:
                for activation in HP_ACTIVATION_UNITS.domain.values:
                    for window_size in HP_DATASET_WINDOW_SIZE.domain.values:
                        for window_shift_ratio in HP_DATASET_WINDOW_SHIFT_RATIO.domain.values:
                            window_shift = int(window_shift_ratio * window_size / 100.0)
                            hparams = {
                                HP_UNITS_1_RATE: units_1,
                                HP_ACTIVATION_UNITS: activation,
                                HP_DATASET_WINDOW_SIZE: window_size,
                                HP_DATASET_WINDOW_SHIFT_RATIO: window_shift_ratio
                            }
                            trail_name = "trail-%d" % session_num
                            logging.info('--- Starting hyperparameter trial: %s' % trail_name)
                            logging.info({h.name: hparams[h] for h in hparams})
                            hyperparameter_summary_writer = tf.summary.create_file_writer(
                                str(hyperparameter_logs_directory / trail_name))

                            train_dataset, validation_dataset, test_dataset, dataset_info = datasets.load(window_size,
                                                                                                          window_shift)

                            input_shape = (dataset_info['window_size'], dataset_info['feature_width'])
                            number_of_classes = dataset_info['number_of_classes']

                            # Create the model with different hyperparameters
                            if model_name == 'simple_rnn':
                                model_architecture = simple_rnn(input_shape=input_shape,
                                                                number_of_classes=number_of_classes,
                                                                sequence_to_label=dataset_info["sequence_to_label"],
                                                                units=hparams[HP_UNITS_1_RATE],
                                                                activation=hparams[HP_ACTIVATION_UNITS])

                            if model_name == 'simple_lstm':
                                model_architecture = simple_lstm(input_shape=input_shape,
                                                                 number_of_classes=number_of_classes,
                                                                 sequence_to_label=dataset_info["sequence_to_label"],
                                                                 units=hparams[HP_UNITS_1_RATE],
                                                                 activation=hparams[HP_ACTIVATION_UNITS])
                            if model_name == 'simple_gru':
                                model_architecture = simple_gru(input_shape=input_shape,
                                                                number_of_classes=number_of_classes,
                                                                sequence_to_label=dataset_info["sequence_to_label"],
                                                                units=hparams[HP_UNITS_1_RATE],
                                                                activation=hparams[HP_ACTIVATION_UNITS])

                            if model_name == 'bidirectional_rnn' and not dataset_info["sequence_to_label"]:
                                model_architecture = bidirectional_rnn(input_shape=input_shape,
                                                                       number_of_classes=number_of_classes,
                                                                       sequence_to_label=dataset_info[
                                                                           "sequence_to_label"],
                                                                       units=hparams[HP_UNITS_1_RATE],
                                                                       activation=hparams[HP_ACTIVATION_UNITS])

                            if model_name == 'bidirectional_lstm' and not dataset_info["sequence_to_label"]:
                                model_architecture = bidirectional_lstm(input_shape=input_shape,
                                                                        number_of_classes=number_of_classes,
                                                                        sequence_to_label=dataset_info[
                                                                            "sequence_to_label"],
                                                                        units=hparams[HP_UNITS_1_RATE],
                                                                        activation=hparams[HP_ACTIVATION_UNITS])
                            if model_name == 'bidirectional_gru' and not dataset_info["sequence_to_label"]:
                                model_architecture = bidirectional_gru(input_shape=input_shape,
                                                                       number_of_classes=number_of_classes,
                                                                       sequence_to_label=dataset_info[
                                                                           "sequence_to_label"],
                                                                       units=hparams[HP_UNITS_1_RATE],
                                                                       activation=hparams[HP_ACTIVATION_UNITS])

                            utils_params.generate_model_directories(model_architecture.name, run_paths)

                            is_ensemble = False
                            is_knowledge_distill = False
                            resume_checkpoint_path = ''
                            trainer = Trainer(model_architecture, is_ensemble,
                                              train_dataset, validation_dataset, dataset_info, run_paths,
                                              resume_checkpoint_path)
                            last_checkpoint = trainer.train()
                            loss, accuracy, completed_epochs = Evaluator(model_architecture, is_ensemble, is_knowledge_distill,
                                                                         last_checkpoint, test_dataset,
                                                                         dataset_info,
                                                                         run_paths).evaluate()

                            # Log the used hyperparameters and results
                            with hyperparameter_summary_writer.as_default():
                                hp.hparams(hparams=hparams)
                                tf.summary.scalar(name=HP_METRIC_LOSS, data=loss,
                                                  step=completed_epochs)  # step is required.
                                tf.summary.scalar(name=HP_METRIC_ACCURACY, data=accuracy,
                                                  step=completed_epochs)  # step is required.

                            session_num += 1

        elif model_name in stacked_models:
            HP_DATASET_WINDOW_SIZE = hp.HParam('dataset_window_size',
                                               hp.Discrete(stacked_two_layer_models_params['dataset_window_size']))
            HP_DATASET_WINDOW_SHIFT_RATIO = hp.HParam('dataset_window_shift',
                                                      hp.Discrete(stacked_two_layer_models_params['dataset_window_shift_ratio']))
            HP_UNITS_1_RATE = hp.HParam(name='units_1',
                                        domain=hp.IntInterval(
                                            min_value=stacked_two_layer_models_params['units_1']['min'],
                                            max_value=stacked_two_layer_models_params['units_1']['max']))
            HP_UNITS_2_RATE = hp.HParam(name='units_2',
                                        domain=hp.IntInterval(
                                            min_value=stacked_two_layer_models_params['units_2']['min'],
                                            max_value=stacked_two_layer_models_params['units_2']['max']))
            HP_ACTIVATION_UNITS = hp.HParam('activation',
                                            hp.Discrete(stacked_two_layer_models_params['activation']))

            # Generate random searches in the HP_UNITS_RATE.domain
            units_1_list = []
            units_n_search = stacked_two_layer_models_params['units_1']['n_search']
            count = 0
            while count < units_n_search:
                units_uniform = HP_UNITS_1_RATE.domain.sample_uniform()
                units_uniform = int(units_uniform)
                if units_uniform not in units_1_list:
                    units_1_list.append(units_uniform)
                    count += 1

            # Generate random searches in the HP_UNITS_RATE.domain
            units_2_list = []
            units_n_search = stacked_two_layer_models_params['units_2']['n_search']
            count = 0
            while count < units_n_search:
                units_uniform = HP_UNITS_2_RATE.domain.sample_uniform()
                units_uniform = int(units_uniform)
                if units_uniform not in units_2_list:
                    units_2_list.append(units_uniform)
                    count += 1

            # Save Tensorboard hyperparameter configuration
            with hyperparameter_summary_writer.as_default():
                hp.hparams_config(
                    hparams=[HP_UNITS_1_RATE, HP_UNITS_2_RATE, HP_ACTIVATION_UNITS,
                             HP_DATASET_WINDOW_SIZE, HP_DATASET_WINDOW_SHIFT_RATIO],
                    metrics=[hp.Metric(tag=HP_METRIC_LOSS, display_name='Loss'),
                             hp.Metric(tag=HP_METRIC_ACCURACY, display_name='Accuracy')]
                )

            session_num = 0

            for units_1 in units_1_list:
                for units_2 in units_2_list:
                    for activation in HP_ACTIVATION_UNITS.domain.values:
                        for window_size in HP_DATASET_WINDOW_SIZE.domain.values:
                            for window_shift_ratio in HP_DATASET_WINDOW_SHIFT_RATIO.domain.values:
                                window_shift = int(window_shift_ratio * window_size / 100.0)
                                hparams = {
                                    HP_UNITS_1_RATE: units_1,
                                    HP_UNITS_2_RATE: units_2,
                                    HP_ACTIVATION_UNITS: activation,
                                    HP_DATASET_WINDOW_SIZE: window_size,
                                    HP_DATASET_WINDOW_SHIFT_RATIO: window_shift_ratio
                                }
                                trail_name = "trail-%d" % session_num
                                logging.info('--- Starting hyperparameter trial: %s' % trail_name)
                                logging.info({h.name: hparams[h] for h in hparams})
                                hyperparameter_summary_writer = tf.summary.create_file_writer(
                                    str(hyperparameter_logs_directory / trail_name))

                                train_dataset, validation_dataset, test_dataset, dataset_info = datasets.load(
                                    window_size,
                                    window_shift)

                                input_shape = (dataset_info['window_size'], dataset_info['feature_width'])
                                number_of_classes = dataset_info['number_of_classes']

                                # Create the model with different hyperparameters
                                if model_name == 'stacked_rnn':
                                    model_architecture = stacked_rnn(input_shape=input_shape,
                                                                     number_of_classes=number_of_classes,
                                                                     sequence_to_label=dataset_info[
                                                                         "sequence_to_label"],
                                                                     units=[hparams[HP_UNITS_1_RATE],
                                                                            hparams[HP_UNITS_2_RATE]],
                                                                     activation=hparams[HP_ACTIVATION_UNITS])

                                if model_name == 'stacked_lstm':
                                    model_architecture = stacked_lstm(input_shape=input_shape,
                                                                      number_of_classes=number_of_classes,
                                                                      sequence_to_label=dataset_info[
                                                                          "sequence_to_label"],
                                                                      units=[hparams[HP_UNITS_1_RATE],
                                                                             hparams[HP_UNITS_2_RATE]],
                                                                      activation=hparams[HP_ACTIVATION_UNITS])
                                if model_name == 'stacked_gru':
                                    model_architecture = stacked_gru(input_shape=input_shape,
                                                                     number_of_classes=number_of_classes,
                                                                     sequence_to_label=dataset_info[
                                                                         "sequence_to_label"],
                                                                     units=[hparams[HP_UNITS_1_RATE],
                                                                            hparams[HP_UNITS_2_RATE]],
                                                                     activation=hparams[HP_ACTIVATION_UNITS])

                                if model_name == 'bidirectional_rnn' and not dataset_info["sequence_to_label"]:
                                    model_architecture = bidirectional_rnn(input_shape=input_shape,
                                                                           number_of_classes=number_of_classes,
                                                                           sequence_to_label=dataset_info[
                                                                               "sequence_to_label"],
                                                                           units=hparams[HP_UNITS_1_RATE],
                                                                           activation=hparams[HP_ACTIVATION_UNITS])

                                if model_name == 'bidirectional_lstm' and not dataset_info["sequence_to_label"]:
                                    model_architecture = bidirectional_lstm(input_shape=input_shape,
                                                                            number_of_classes=number_of_classes,
                                                                            sequence_to_label=dataset_info[
                                                                                "sequence_to_label"],
                                                                            units=hparams[HP_UNITS_1_RATE],
                                                                            activation=hparams[HP_ACTIVATION_UNITS])
                                if model_name == 'bidirectional_gru' and not dataset_info["sequence_to_label"]:
                                    model_architecture = bidirectional_gru(input_shape=input_shape,
                                                                           number_of_classes=number_of_classes,
                                                                           sequence_to_label=dataset_info[
                                                                               "sequence_to_label"],
                                                                           units=hparams[HP_UNITS_1_RATE],
                                                                           activation=hparams[HP_ACTIVATION_UNITS])

                                utils_params.generate_model_directories(model_architecture.name, run_paths)

                                is_ensemble = False
                                is_knowledge_distill = False
                                resume_checkpoint_path = ''
                                trainer = Trainer(model_architecture, is_ensemble,
                                                  train_dataset, validation_dataset, dataset_info, run_paths,
                                                  resume_checkpoint_path)
                                last_checkpoint = trainer.train()
                                loss, accuracy, completed_epochs = Evaluator(model_architecture, is_ensemble,
                                                                             is_knowledge_distill,
                                                                             last_checkpoint, test_dataset,
                                                                             dataset_info,
                                                                             run_paths).evaluate()

                                # Log the used hyperparameters and results
                                with hyperparameter_summary_writer.as_default():
                                    hp.hparams(hparams=hparams)
                                    tf.summary.scalar(name=HP_METRIC_LOSS, data=loss,
                                                      step=completed_epochs)  # step is required.
                                    tf.summary.scalar(name=HP_METRIC_ACCURACY, data=accuracy,
                                                      step=completed_epochs)  # step is required.

                                session_num += 1
