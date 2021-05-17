import io
import os
import shutil
import gin
import tensorflow as tf
import logging
import pathlib
import matplotlib.pyplot as plt
import enum

from evaluation.metrics import (ConfusionMatrix, ClassWiseMetric, AverageMetric, Metric,
                                convert_AverageMetric_result, convert_ClassWiseMetric_result,
                                format_metric_output, format_confusion_matrix_metric_output, )

training_logger = logging.getLogger('training')

EPOCH_COUNTER_PRINT_INTERVAL = 500

Early_stopping_type = enum.Enum('Early_stopping_type', 'Relative Average')


@gin.configurable
class Trainer(object):
    '''Train the given model on the given dataset.'''

    def __init__(self, model, is_ensemble, training_dataset, validation_dataset, dataset_info, run_paths,
                 resume_checkpoint,
                 epochs, log_interval, checkpoint_interval, learning_rate, early_stopping_delay,  # <- configs
                 patience, minimum_delta, early_stopping_test_interval, early_stopping_trigger_type,
                 early_stopping_metric):  # <- configs

        model_directory = run_paths['model_directories'][model.name]

        self.model = model
        self.is_ensemble = is_ensemble

        self.train_dataset = training_dataset
        self.validation_dataset = validation_dataset

        self.epochs = epochs
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.epoch_counter = tf.Variable(initial_value=0, trainable=False)

        # Summary Writer
        self.train_summary_writer = tf.summary.create_file_writer(logdir=str(model_directory['summaries'] / 'train'))
        self.validation_summary_writer = tf.summary.create_file_writer(
            logdir=str(model_directory['summaries'] / 'validation'))

        # Loss objective
        # Sparse means that the target labels are integers, not one-hot encoded arrays.
        # from_logits: If True then the input is expected to be not normalised,
        #   if a softmax activation function is used in the last layer, then the output is normalised,
        #   therefore from_logits is set to False.
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.num_classes = dataset_info['number_of_classes']  # Number of classes in the output

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_confusion_matrix_metrics = ConfusionMatrix(num_classes=self.num_classes,
                                                              sequence_to_label=dataset_info['sequence_to_label'],
                                                              name='train_confusion_matrix_metrics')
        self.train_class_wise_metrics = ClassWiseMetric(num_classes=self.num_classes,
                                                        sequence_to_label=dataset_info['sequence_to_label'],
                                                        name='train_class_wise_metrics')
        self.train_average_metrics = AverageMetric(num_classes=self.num_classes,
                                                   sequence_to_label=dataset_info['sequence_to_label'],
                                                   name='train_average_metrics')

        self.validation_loss = tf.keras.metrics.Mean(name='validation_loss')
        self.validation_confusion_matrix_metrics = ConfusionMatrix(num_classes=self.num_classes,
                                                                   sequence_to_label=dataset_info['sequence_to_label'],
                                                                   name='validation_confusion_matrix_metrics')
        self.validation_class_wise_metrics = ClassWiseMetric(num_classes=self.num_classes,
                                                             sequence_to_label=dataset_info['sequence_to_label'],
                                                             name='validation_class_wise_metrics')
        self.validation_average_metrics = AverageMetric(num_classes=self.num_classes,
                                                        sequence_to_label=dataset_info['sequence_to_label'],
                                                        name='validation_average_metrics')

        # Checkpoint Manager
        checkpoint_directory = model_directory['training_checkpoints']
        checkpoint = tf.train.Checkpoint(step_counter=self.epoch_counter, optimizer=self.optimizer, model=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint, max_to_keep=3,
                                                             directory=checkpoint_directory)
        if resume_checkpoint:
            if pathlib.Path(resume_checkpoint).is_dir():
                resume_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=resume_checkpoint)
            # Raises an exception if any existing Python objects in the dependency graph are unmatched.
            checkpoint.restore(resume_checkpoint).assert_existing_objects_matched()
            print(f"Restored from checkpoint: '{resume_checkpoint}'.")
        else:
            print('No checkpoint given, initializing from scratch.')

        # Saved Model for the run
        self.saved_model_for_this_run_path = run_paths['saved_models_directory'] / self.model.name

        # Saved Model as a whole
        self.trained_model_path = run_paths['trained_models_directory'] / self.model.name

        # Early-stopping
        self.patience = patience
        self.minimum_delta = minimum_delta
        self.early_stopping_delay = early_stopping_delay
        self.early_stopping_test_interval = early_stopping_test_interval
        self.early_stopping_trigger_type = Early_stopping_type(early_stopping_trigger_type + 1)
        self.early_stopping_metric = Metric.from_string(early_stopping_metric)
        early_stopping_checkpoint = tf.train.Checkpoint(step_counter=self.epoch_counter, model=self.model)
        self.early_stopping_checkpoint_manager = tf.train.CheckpointManager(checkpoint=early_stopping_checkpoint,
                                                                            max_to_keep=1,
                                                                            directory=str(model_directory[
                                                                                              'early_stopping_checkpoints']))
        self.last_successful_metric = self.best_metric = self.epochs_without_improvement = 0
        self.last_metrics = list()

        if self.early_stopping_test_interval > 0:
            logging.info(
                f'Early-stopping enabled, patience: {patience}, delta: {minimum_delta}, metric: {early_stopping_metric}.')
        else:
            logging.info('Early-stopping disabled.')

    @tf.function
    def train_step(self, x_batch_timestep_n_features, x_batch_labels):
        '''Test the model by feeding it a batch of features with 'training' set to False.
            Also update the validation metrics.

            Parameters
            ----------
            x_batch_timestep_n_features : tf.tensor
                A batch of timestep, features of the data

            x_batch_labels : tf.tensor
                A batch of labels.
        '''
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(x_batch_timestep_n_features, training=True)
            # # Add asserts to check the shape of the output.
            # tf.debugging.assert_equal(predictions.shape, x_batch_labels.shape)
            loss = self.loss_object(x_batch_labels, predictions)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_confusion_matrix_metrics.update_state(x_batch_labels, predictions)
        self.train_class_wise_metrics.update_state(x_batch_labels, predictions)
        self.train_average_metrics.update_state(x_batch_labels, predictions)

    @tf.function
    def validation_step(self, x_batch_timestep_n_features, x_batch_label, early_stopping=False) -> None:
        '''Test the model by feeding it a batch of features with 'training' set to False.
        Also updates the validation metrics.
        
        Parameters
        ----------               
        x_batch_timestep_n_features : tf.tensor
            A batch of images.
        
        x_batch_label : tf.tensor
            A batch of labels.

        early_stopping : boolean
            Whether to update all of the metrics or just the AverageMetric for testing early-stopping.
        '''
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(x_batch_timestep_n_features, training=False)
        validation_loss = self.loss_object(x_batch_label, predictions)

        if not early_stopping:
            self.validation_loss(validation_loss)
            self.validation_confusion_matrix_metrics.update_state(x_batch_label, predictions)
            self.validation_class_wise_metrics.update_state(x_batch_label, predictions)

        self.validation_average_metrics.update_state(x_batch_label, predictions)

    def test_early_stopping(self) -> bool:
        '''Test for early-stopping with the validation dataset.

        Two types of early-stopping tests are implemented:
        - Relative: Improve by a given value (`minimum_delta`) in a given number of epochs (`patience`).
        - Average: The average improvement has to reach a target value (`minimum_delta`) in a given number of epochs (`patience`).

        Returns
        -------
        bool
            True if early-stopping is triggered, False if not.
        '''
        self.validation_average_metrics.reset_states()
        for val_batch_timestep_n_features, val_batch_labels in self.validation_dataset:
            self.validation_step(val_batch_timestep_n_features, val_batch_labels, early_stopping=True)

        current_metric = self.validation_average_metrics.calculate_macro_metric(self.early_stopping_metric)

        early_stopping_criteria_satisfied = True
        if self.early_stopping_trigger_type == Early_stopping_type.Relative:
            if (current_metric - self.last_successful_metric) > self.minimum_delta:
                early_stopping_criteria_satisfied = False


        elif self.early_stopping_trigger_type == Early_stopping_type.Average:
            self.last_metrics.append(current_metric - self.last_successful_metric)

            if (sum(self.last_metrics) / len(self.last_metrics)) > self.minimum_delta:
                early_stopping_criteria_satisfied = False
                self.last_metrics = list()

        if current_metric > self.best_metric:  # Always save the currently best model.
            self.best_metric = current_metric
            self.early_stopping_checkpoint_manager.save()

        if early_stopping_criteria_satisfied:
            self.epochs_without_improvement += self.early_stopping_test_interval

            if self.epochs_without_improvement > self.patience:
                return True

        else:
            self.last_successful_metric = current_metric
            self.epochs_without_improvement = 0

        return False

    def train(self):
        training_logger.info(f"Start training for model '{self.model.name}' for {self.epochs:,} epochs")

        # Reset the Train Metrics
        self.train_loss.reset_states()
        self.train_confusion_matrix_metrics.reset_states()
        self.train_class_wise_metrics.reset_states()
        self.train_average_metrics.reset_states()

        # Iterate over the batches of the dataset.
        for x_batch_timestep_n_features, x_batch_labels in self.train_dataset:

            epoch_counter = int(self.epoch_counter.assign_add(1))

            if self.is_ensemble:
                x_batch_timestep_n_features = [x_batch_timestep_n_features for _ in range(len(self.model.input))]

            if epoch_counter % EPOCH_COUNTER_PRINT_INTERVAL == 0:
                tf.print(f"Epoch: {epoch_counter} / {self.epochs} ===>")

            self.train_step(x_batch_timestep_n_features, x_batch_labels)

            if epoch_counter % self.log_interval == 0:
                # Reset the Validation Metrics
                self.validation_loss.reset_states()
                self.validation_confusion_matrix_metrics.reset_states()
                self.validation_class_wise_metrics.reset_states()
                self.validation_average_metrics.reset_states()

                for val_batch_timestep_n_features, val_batch_labels in self.validation_dataset:
                    if self.is_ensemble:
                        val_batch_timestep_n_features = [val_batch_timestep_n_features for _ in
                                                         range(len(self.model.input))]
                    self.validation_step(val_batch_timestep_n_features, val_batch_labels)

                self.logging_and_summaries(epoch_counter)

            if (self.early_stopping_test_interval > 0) and (epoch_counter > self.early_stopping_delay) and (
                    (epoch_counter % self.early_stopping_test_interval) == 0):
                if self.test_early_stopping():
                    logging.info(f'Early-stopping is triggered at epoch {epoch_counter}.')
                    early_stopping_checkpoint = self._on_epoch_end(early_stopping_triggered=True)
                    return early_stopping_checkpoint

            if epoch_counter % self.checkpoint_interval == 0:  # Save checkpoint
                checkpoint_path = self.checkpoint_manager.save()
                logging.info(f"Saving checkpoint: '{checkpoint_path}'")

            if epoch_counter % self.epochs == 0:  # Save final checkpoint
                training_logger.info(f"Finished training after {epoch_counter} epochs.")
                last_checkpoint = self._on_epoch_end(early_stopping_triggered=False)
                return last_checkpoint

    def _on_epoch_end(self, early_stopping_triggered=False) -> str:
        last_checkpoint = None
        try:
            if early_stopping_triggered:
                early_stopping_checkpoint = self.early_stopping_checkpoint_manager.latest_checkpoint
                tf.train.Checkpoint(model=self.model).restore(save_path=early_stopping_checkpoint).expect_partial()

            last_checkpoint = self.checkpoint_manager.save()

            training_logger.info(f"Last checkpoint: '{last_checkpoint}'")
        except:
            logging.error(f"Failed to save last checkpoint")

        try:
            # Save final model for this run
            tf.train.Checkpoint(model=self.model).save(file_prefix=str(self.saved_model_for_this_run_path))
            logging.info(f"Final model for this run saved at '{self.saved_model_for_this_run_path}'.")
        except:
            logging.error(f"Failed to save final model for this run at '{self.saved_model_for_this_run_path}'.")

        if self.trained_model_path.exists():
            # deleting previous trained model files that were saved
            for filename in os.listdir(str(self.trained_model_path)):
                file_path = str(self.trained_model_path / filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logging.info(f'Failed to delete {file_path}; Reason: {e}.')
        else:
            self.trained_model_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save final model as a whole
            self.trained_model_path = self.trained_model_path / self.model.name
            tf.train.Checkpoint(model=self.model).save(file_prefix=str(self.trained_model_path))
            logging.info(f"Trained model saved at '{self.trained_model_path}'.")
        except:
            logging.error(f"Failed to save trained model at '{self.trained_model_path}'.")

        return last_checkpoint

    def logging_and_summaries(self, epoch_counter) -> None:
        train_loss_result = self.train_loss.result()
        train_confusion_matrix_results = self.train_confusion_matrix_metrics.result()

        train_class_wise_metric_results = self.train_class_wise_metrics.result()
        train_class_wise_metric_results = convert_ClassWiseMetric_result(train_class_wise_metric_results,
                                                                         self.num_classes)
        train_class_wise_metric_results_string = format_metric_output(train_class_wise_metric_results)

        train_average_metric_results = self.train_average_metrics.result()
        train_average_metric_results = convert_AverageMetric_result(train_average_metric_results)
        train_average_metric_results_string = format_metric_output(train_average_metric_results)

        validation_loss_result = self.validation_loss.result()
        validation_confusion_matrix_results = self.validation_confusion_matrix_metrics.result()

        validation_class_wise_metric_results = self.validation_class_wise_metrics.result()
        validation_class_wise_metric_results = convert_ClassWiseMetric_result(validation_class_wise_metric_results,
                                                                              self.num_classes)
        validation_class_wise_metric_results_string = format_metric_output(validation_class_wise_metric_results)

        validation_average_metric_results = self.validation_average_metrics.result()
        validation_average_metric_results = convert_AverageMetric_result(validation_average_metric_results)
        validation_average_metric_results_string = format_metric_output(validation_average_metric_results)

        template = ('Step {} ->, \n'
                    'Train Loss: {:.3f}, \n'
                    'Train Class Wise Metric: {}, \n'
                    'Train Average Metric: {}, \n'
                    'Validation Loss: {:.3f}, \n'
                    'Validation Class Wise Metric: {}, \n'
                    'Validation Average Metric: {} \n'
                    '*******************************************')
        training_logger.info(template.format(epoch_counter,
                                             train_loss_result,
                                             train_class_wise_metric_results_string,
                                             train_average_metric_results_string,
                                             validation_loss_result,
                                             validation_class_wise_metric_results_string,
                                             validation_average_metric_results_string))
        training_logger.info(train_confusion_matrix_results)
        training_logger.info(validation_confusion_matrix_results)
        training_logger.info("//////////////////////////////////////////////")

        train_confusion_matrix_fig = format_confusion_matrix_metric_output(self.num_classes,
                                                                           train_confusion_matrix_results)

        # Write summary to tensorboard
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss_result, step=epoch_counter)

            for name, value in train_class_wise_metric_results.items():
                tf.summary.scalar(name, value, step=epoch_counter)

            for name, value in train_average_metric_results.items():
                tf.summary.scalar(name, value, step=epoch_counter)

            tf.summary.image(f'Training confusion matrix',
                             self._plot_to_image(train_confusion_matrix_fig), step=epoch_counter)

        validation_confusion_matrix_fig = format_confusion_matrix_metric_output(self.num_classes,
                                                                                validation_confusion_matrix_results)

        with self.validation_summary_writer.as_default():
            tf.summary.scalar('loss', validation_loss_result, step=epoch_counter)

            for name, value in validation_class_wise_metric_results.items():
                tf.summary.scalar(name, value, step=epoch_counter)

            for name, value in validation_average_metric_results.items():
                tf.summary.scalar(name, value, step=epoch_counter)

            # f'Confusion matrix after {epoch_counter} epochs'
            tf.summary.image(f'Validation confusion matrix',
                             self._plot_to_image(validation_confusion_matrix_fig), step=epoch_counter)

    def _plot_to_image(self, figure):
        '''Converts the matplotlib plot specified by 'figure' to a PNG image and returns it.
        The supplied figure is closed and inaccessible after this call.'''

        buf = io.BytesIO()

        # Use plt.savefig to save the plot to a PNG in memory.
        plt.savefig(buf, format='png')

        # Closing the figure prevents it from being displayed directly inside the notebook.
        plt.close(figure)
        buf.seek(0)

        # Use tf.image.decode_png to convert the PNG buffer
        # to a TF image. Make sure you use 4 channels.
        image = tf.image.decode_png(buf.getvalue(), channels=4)

        # Use tf.expand_dims to add the batch dimension
        image = tf.expand_dims(image, 0)

        return image


@gin.configurable
class KDTrainer(object):
    def __init__(self, teacher_model, student_model, training_dataset, validation_dataset, dataset_info, run_paths,
                 resume_checkpoint,
                 epochs, log_interval, checkpoint_interval, learning_rate,
                 temperature, alpha, beta,  # <- configs
                 early_stopping_delay, patience, minimum_delta,  # <- configs
                 early_stopping_test_interval, early_stopping_trigger_type,  # <- configs
                 early_stopping_metric):  # <- configs

        model_directory = run_paths['model_directories'][student_model.name]

        self.teacher_model = teacher_model
        self.student_model = student_model

        self.train_dataset = training_dataset
        self.validation_dataset = validation_dataset

        self.epochs = epochs
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.epoch_counter = tf.Variable(initial_value=0, trainable=False)

        # Summary Writer
        self.train_summary_writer = tf.summary.create_file_writer(logdir=str(model_directory['summaries'] / 'train'))
        self.validation_summary_writer = tf.summary.create_file_writer(
            logdir=str(model_directory['summaries'] / 'validation'))

        # Loss objective
        self.train_distillation_loss_object = self._get_kd_loss  # Ref to the loss function
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.normal_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        self.num_classes = dataset_info['number_of_classes']  # Number of classes in the output

        # Metrics
        self.train_distillation_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_confusion_matrix_metrics = ConfusionMatrix(num_classes=self.num_classes,
                                                              sequence_to_label=dataset_info['sequence_to_label'],
                                                              name='train_confusion_matrix_metrics')
        self.train_class_wise_metrics = ClassWiseMetric(num_classes=self.num_classes,
                                                        sequence_to_label=dataset_info['sequence_to_label'],
                                                        name='train_class_wise_metrics')
        self.train_average_metrics = AverageMetric(num_classes=self.num_classes,
                                                   sequence_to_label=dataset_info['sequence_to_label'],
                                                   name='train_average_metrics')

        self.validation_loss = tf.keras.metrics.Mean(name='validation_loss')
        self.validation_confusion_matrix_metrics = ConfusionMatrix(num_classes=self.num_classes,
                                                                   sequence_to_label=dataset_info['sequence_to_label'],
                                                                   name='validation_confusion_matrix_metrics')
        self.validation_class_wise_metrics = ClassWiseMetric(num_classes=self.num_classes,
                                                             sequence_to_label=dataset_info['sequence_to_label'],
                                                             name='validation_class_wise_metrics')
        self.validation_average_metrics = AverageMetric(num_classes=self.num_classes,
                                                        sequence_to_label=dataset_info['sequence_to_label'],
                                                        name='validation_average_metrics')

        # Checkpoint Manager
        checkpoint_directory = model_directory['training_checkpoints']
        checkpoint = tf.train.Checkpoint(step_counter=self.epoch_counter, optimizer=self.optimizer,
                                         model=self.student_model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint, max_to_keep=3,
                                                             directory=checkpoint_directory)
        if resume_checkpoint:
            if pathlib.Path(resume_checkpoint).is_dir():
                resume_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=resume_checkpoint)
            # Raises an exception if any existing Python objects in the dependency graph are unmatched.
            checkpoint.restore(resume_checkpoint).assert_existing_objects_matched()
            print(f"Restored from checkpoint: '{resume_checkpoint}'.")
        else:
            print('No checkpoint given, initializing from scratch.')

        # Saved Model for the run
        self.saved_model_for_this_run_path = run_paths['saved_models_directory'] / self.student_model.name

        # Saved Model as a whole
        self.trained_model_path = run_paths['trained_models_directory'] / self.student_model.name

        # Early-stopping
        self.patience = patience
        self.minimum_delta = minimum_delta
        self.early_stopping_delay = early_stopping_delay
        self.early_stopping_test_interval = early_stopping_test_interval
        self.early_stopping_trigger_type = Early_stopping_type(early_stopping_trigger_type + 1)
        self.early_stopping_metric = Metric.from_string(early_stopping_metric)
        early_stopping_checkpoint = tf.train.Checkpoint(step_counter=self.epoch_counter, model=self.student_model)
        self.early_stopping_checkpoint_manager = tf.train.CheckpointManager(checkpoint=early_stopping_checkpoint,
                                                                            max_to_keep=1,
                                                                            directory=str(model_directory[
                                                                                              'early_stopping_checkpoints']))
        self.last_successful_metric = self.best_metric = self.epochs_without_improvement = 0
        self.last_metrics = list()

        if self.early_stopping_test_interval > 0:
            logging.info(
                f'Early-stopping enabled, patience: {patience}, delta: {minimum_delta}, metric: {early_stopping_metric}.')
        else:
            logging.info('Early-stopping disabled.')

    @tf.function
    def train_step(self, x_batch_timestep_n_features, x_batch_labels):
        '''Test the model by feeding it a batch of features with 'training' set to False.
            Also update the validation metrics.

            Parameters
            ----------
            x_batch_timestep_n_features : tf.tensor
                A batch of timestep, features of the data

            x_batch_labels : tf.tensor
                A batch of labels.
        '''
        teacher_logits = self.teacher_model(x_batch_timestep_n_features, training=False)

        with tf.GradientTape() as tape:
            student_logits = self.student_model(x_batch_timestep_n_features, training=True)
            # # Add asserts to check the shape of the output.
            # tf.debugging.assert_equal(predictions.shape, x_batch_labels.shape)
            distillation_loss = self.train_distillation_loss_object(student_logits, teacher_logits,
                                                                    x_batch_labels, self.temperature,
                                                                    self.alpha, self.beta)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        gradients = tape.gradient(distillation_loss, self.student_model.trainable_variables)

        # As mentioned in Section 2 of https://arxiv.org/abs/1503.02531
        gradients = [gradient * (self.temperature ** 2) for gradient in gradients]

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))

        predictions = tf.nn.softmax(student_logits)
        train_loss = self.normal_loss_object(x_batch_labels, predictions)
        self.train_distillation_loss(distillation_loss)
        self.train_loss(train_loss)
        self.train_confusion_matrix_metrics.update_state(x_batch_labels, predictions)
        self.train_class_wise_metrics.update_state(x_batch_labels, predictions)
        self.train_average_metrics.update_state(x_batch_labels, predictions)

    @tf.function
    def validation_step(self, x_batch_timestep_n_features, x_batch_label, early_stopping=False) -> None:
        '''Test the model by feeding it a batch of features with 'training' set to False.
        Also updates the validation metrics.

        Parameters
        ----------
        x_batch_timestep_n_features : tf.tensor
            A batch of images.

        x_batch_label : tf.tensor
            A batch of labels.

        early_stopping : boolean
            Whether to update all of the metrics or just the AverageMetric for testing early-stopping.
        '''
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        student_logits = self.student_model(x_batch_timestep_n_features, training=False)
        predictions = tf.nn.softmax(student_logits)
        validation_loss = self.normal_loss_object(x_batch_label, predictions)

        if not early_stopping:
            self.validation_loss(validation_loss)
            self.validation_confusion_matrix_metrics.update_state(x_batch_label, predictions)
            self.validation_class_wise_metrics.update_state(x_batch_label, predictions)

        self.validation_average_metrics.update_state(x_batch_label, predictions)

    def test_early_stopping(self) -> bool:
        '''Test for early-stopping with the validation dataset.

        Two types of early-stopping tests are implemented:
        - Relative: Improve by a given value (`minimum_delta`) in a given number of epochs (`patience`).
        - Average: The average improvement has to reach a target value (`minimum_delta`) in a given number of epochs (`patience`).

        Returns
        -------
        bool
            True if early-stopping is triggered, False if not.
        '''
        self.validation_average_metrics.reset_states()
        for val_batch_timestep_n_features, val_batch_labels in self.validation_dataset:
            self.validation_step(val_batch_timestep_n_features, val_batch_labels, early_stopping=True)

        current_metric = self.validation_average_metrics.calculate_macro_metric(self.early_stopping_metric)

        early_stopping_criteria_satisfied = True
        if self.early_stopping_trigger_type == Early_stopping_type.Relative:
            if (current_metric - self.last_successful_metric) > self.minimum_delta:
                early_stopping_criteria_satisfied = False


        elif self.early_stopping_trigger_type == Early_stopping_type.Average:
            self.last_metrics.append(current_metric - self.last_successful_metric)

            if (sum(self.last_metrics) / len(self.last_metrics)) > self.minimum_delta:
                early_stopping_criteria_satisfied = False
                self.last_metrics = list()

        if current_metric > self.best_metric:  # Always save the currently best model.
            self.best_metric = current_metric
            self.early_stopping_checkpoint_manager.save()

        if early_stopping_criteria_satisfied:
            self.epochs_without_improvement += self.early_stopping_test_interval

            if self.epochs_without_improvement > self.patience:
                return True

        else:
            self.last_successful_metric = current_metric
            self.epochs_without_improvement = 0

        return False

    def train(self):
        training_logger.info(f"Start training for model '{self.student_model.name}' for {self.epochs:,} epochs")

        # Reset the metrics
        self.train_distillation_loss.reset_states()
        self.train_loss.reset_states()
        self.train_confusion_matrix_metrics.reset_states()
        self.train_class_wise_metrics.reset_states()
        self.train_average_metrics.reset_states()

        # Iterate over the batches of the dataset.
        for x_batch_timestep_n_features, x_batch_labels in self.train_dataset:

            epoch_counter = int(self.epoch_counter.assign_add(1))

            if epoch_counter % EPOCH_COUNTER_PRINT_INTERVAL == 0:
                tf.print(f"Epoch: {epoch_counter} / {self.epochs} ===>")

            self.train_step(x_batch_timestep_n_features, x_batch_labels)

            if epoch_counter % self.log_interval == 0:
                # Reset the Validation Metrics
                self.validation_loss.reset_states()
                self.validation_confusion_matrix_metrics.reset_states()
                self.validation_class_wise_metrics.reset_states()
                self.validation_average_metrics.reset_states()

                for val_batch_timestep_n_features, val_batch_labels in self.validation_dataset:
                    self.validation_step(val_batch_timestep_n_features, val_batch_labels)

                self.logging_and_summaries(epoch_counter)

            if (self.early_stopping_test_interval > 0) and (epoch_counter > self.early_stopping_delay) and (
                    (epoch_counter % self.early_stopping_test_interval) == 0):
                if self.test_early_stopping():
                    logging.info(f'Early-stopping is triggered at epoch {epoch_counter}.')
                    early_stopping_checkpoint = self._on_epoch_end(early_stopping_triggered=True)
                    return early_stopping_checkpoint

            if epoch_counter % self.checkpoint_interval == 0:  # Save checkpoint
                checkpoint_path = self.checkpoint_manager.save()
                logging.info(f"Saving checkpoint: '{checkpoint_path}'")

            if epoch_counter % self.epochs == 0:  # Save final checkpoint
                training_logger.info(f"Finished training after {epoch_counter} epochs.")
                last_checkpoint = self._on_epoch_end(early_stopping_triggered=False)
                return last_checkpoint

    def _on_epoch_end(self, early_stopping_triggered=False) -> str:
        last_checkpoint = None
        try:
            if early_stopping_triggered:
                early_stopping_checkpoint = self.early_stopping_checkpoint_manager.latest_checkpoint
                tf.train.Checkpoint(model=self.student_model).restore(
                    save_path=early_stopping_checkpoint).expect_partial()

            last_checkpoint = self.checkpoint_manager.save()

            training_logger.info(f"Last checkpoint: '{last_checkpoint}'")
        except:
            logging.error(f"Failed to save last checkpoint")

        try:
            # Save final model for this run
            tf.train.Checkpoint(model=self.student_model).save(file_prefix=str(self.saved_model_for_this_run_path))
            logging.info(f"Final model for this run saved at '{self.saved_model_for_this_run_path}'.")
        except:
            logging.error(f"Failed to save final model for this run at '{self.saved_model_for_this_run_path}'.")

        if self.trained_model_path.exists():
            # deleting previous trained model files that were saved
            for filename in os.listdir(str(self.trained_model_path)):
                file_path = str(self.trained_model_path / filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logging.info(f'Failed to delete {file_path}; Reason: {e}.')
        else:
            self.trained_model_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save final model as a whole
            self.trained_model_path = self.trained_model_path / self.student_model.name
            tf.train.Checkpoint(model=self.student_model).save(file_prefix=str(self.trained_model_path))
            logging.info(f"Trained model saved at '{self.trained_model_path}'.")
        except:
            logging.error(f"Failed to save trained model at '{self.trained_model_path}'.")

        return last_checkpoint

    def logging_and_summaries(self, epoch_counter) -> None:
        train_distillation_loss_result = self.train_distillation_loss.result()
        train_loss_result = self.train_loss.result()
        train_confusion_matrix_results = self.train_confusion_matrix_metrics.result()

        train_class_wise_metric_results = self.train_class_wise_metrics.result()
        train_class_wise_metric_results = convert_ClassWiseMetric_result(train_class_wise_metric_results,
                                                                         self.num_classes)
        train_class_wise_metric_results_string = format_metric_output(train_class_wise_metric_results)

        train_average_metric_results = self.train_average_metrics.result()
        train_average_metric_results = convert_AverageMetric_result(train_average_metric_results)
        train_average_metric_results_string = format_metric_output(train_average_metric_results)

        validation_loss_result = self.validation_loss.result()
        validation_confusion_matrix_results = self.validation_confusion_matrix_metrics.result()

        validation_class_wise_metric_results = self.validation_class_wise_metrics.result()
        validation_class_wise_metric_results = convert_ClassWiseMetric_result(validation_class_wise_metric_results,
                                                                              self.num_classes)
        validation_class_wise_metric_results_string = format_metric_output(validation_class_wise_metric_results)

        validation_average_metric_results = self.validation_average_metrics.result()
        validation_average_metric_results = convert_AverageMetric_result(validation_average_metric_results)
        validation_average_metric_results_string = format_metric_output(validation_average_metric_results)

        template = ('Step {} ->, \n'
                    'Train Distillation Loss: {:.3f}, \n'
                    'Train Loss: {:.3f}, \n'
                    'Train Class Wise Metric: {}, \n'
                    'Train Average Metric: {}, \n'
                    'Validation Loss: {:.3f}, \n'
                    'Validation Class Wise Metric: {}, \n'
                    'Validation Average Metric: {} \n'
                    '*******************************************')
        training_logger.info(template.format(epoch_counter,
                                             train_distillation_loss_result,
                                             train_loss_result,
                                             train_class_wise_metric_results_string,
                                             train_average_metric_results_string,
                                             validation_loss_result,
                                             validation_class_wise_metric_results_string,
                                             validation_average_metric_results_string))
        training_logger.info(train_confusion_matrix_results)
        training_logger.info(validation_confusion_matrix_results)
        training_logger.info("//////////////////////////////////////////////")

        train_confusion_matrix_fig = format_confusion_matrix_metric_output(self.num_classes,
                                                                           train_confusion_matrix_results)

        # Write summary to tensorboard
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', train_distillation_loss_result, step=epoch_counter)
            tf.summary.scalar('loss', train_loss_result, step=epoch_counter)

            for name, value in train_class_wise_metric_results.items():
                tf.summary.scalar(name, value, step=epoch_counter)

            for name, value in train_average_metric_results.items():
                tf.summary.scalar(name, value, step=epoch_counter)

            tf.summary.image(f'Training confusion matrix',
                             self._plot_to_image(train_confusion_matrix_fig), step=epoch_counter)

        validation_confusion_matrix_fig = format_confusion_matrix_metric_output(self.num_classes,
                                                                                validation_confusion_matrix_results)

        with self.validation_summary_writer.as_default():
            tf.summary.scalar('loss', validation_loss_result, step=epoch_counter)

            for name, value in validation_class_wise_metric_results.items():
                tf.summary.scalar(name, value, step=epoch_counter)

            for name, value in validation_average_metric_results.items():
                tf.summary.scalar(name, value, step=epoch_counter)

            # f'Confusion matrix after {epoch_counter} epochs'
            tf.summary.image(f'Validation confusion matrix',
                             self._plot_to_image(validation_confusion_matrix_fig), step=epoch_counter)

    def _plot_to_image(self, figure):
        '''Converts the matplotlib plot specified by 'figure' to a PNG image and returns it.
        The supplied figure is closed and inaccessible after this call.'''

        buf = io.BytesIO()

        # Use plt.savefig to save the plot to a PNG in memory.
        plt.savefig(buf, format='png')

        # Closing the figure prevents it from being displayed directly inside the notebook.
        plt.close(figure)
        buf.seek(0)

        # Use tf.image.decode_png to convert the PNG buffer
        # to a TF image. Make sure you use 4 channels.
        image = tf.image.decode_png(buf.getvalue(), channels=4)

        # Use tf.expand_dims to add the batch dimension
        image = tf.expand_dims(image, 0)

        return image

    def _get_kd_loss(self, student_logits, teacher_logits,
                     true_labels, temperature,
                     alpha, beta):
        teacher_probs = tf.nn.softmax(teacher_logits / temperature)
        kd_loss = tf.keras.losses.categorical_crossentropy(
            teacher_probs, student_logits / temperature,
            from_logits=True)

        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(
            true_labels, student_logits, from_logits=True)

        total_loss = (alpha * kd_loss) + (beta * ce_loss)
        return total_loss / (alpha + beta)
