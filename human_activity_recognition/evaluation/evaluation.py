import tensorflow as tf
import datetime
import pathlib
import logging
import seaborn as sb
import matplotlib.pyplot as plt
import io

from evaluation.metrics import ConfusionMatrix, ClassWiseMetric, AverageMetric, \
                               convert_AverageMetric_result, convert_ClassWiseMetric_result, \
                               format_confusion_matrix_metric_output, format_metric_output

evaluation_logger = logging.getLogger('evaluation')


class Evaluator:

    def __init__(self, model, is_ensemble, is_knowledge_distill, checkpoint_path, test_dataset, dataset_info, run_paths,
                 restore_from_checkpoint=True):

        model_directory = run_paths['model_directories'][model.name]

        self.test_dataset = test_dataset
        self.model = model
        self.is_ensemble = is_ensemble
        self.is_knowledge_distill = is_knowledge_distill

        # Summary Writer
        self.test_summary_writer = tf.summary.create_file_writer(logdir=str(model_directory['summaries'] / 'test'))

        # Metrics
        self.num_classes = dataset_info['number_of_classes']
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_confusion_matrix_metrics = ConfusionMatrix(num_classes=self.num_classes,
                                                             sequence_to_label=dataset_info['sequence_to_label'],
                                                             name='test_confusion_matrix_metrics')
        self.test_class_wise_metrics = ClassWiseMetric(num_classes=self.num_classes,
                                                       sequence_to_label=dataset_info['sequence_to_label'],
                                                       name='test_class_wise_metrics')
        self.test_average_metrics = AverageMetric(num_classes=self.num_classes,
                                                  sequence_to_label=dataset_info['sequence_to_label'],
                                                  name='test_average_metrics')

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        # Load model from checkpoint
        step_counter = tf.Variable(initial_value=0, trainable=False)
        evaluation_checkpoint = tf.train.Checkpoint(step_counter=step_counter, model=self.model)

        # if restore_from_checkpoint is true, then restore from the checkpoints
        # else just evaluate on the given model in parameters 'model'.
        # #
        # restore_from_checkpoint = False makes sense when you already have loaded
        # the model before calling the Evaluator class.
        if restore_from_checkpoint:
            # TODO: commenting checkpoint because it is not working when training multiple models in a loop from main
            if checkpoint_path:
                # Only the model is loaded, therefore 'assert_existing_objects_matched' is called
                # instead of 'assert_consumed', which tests all variables.
                # Use 'expect_partial' to silence warnings.
                evaluation_checkpoint.restore(checkpoint_path).expect_partial()
            else:
                checkpoint_manager = tf.train.CheckpointManager(checkpoint=evaluation_checkpoint, max_to_keep=3,
                                                                directory=str(model_directory['training_checkpoints']))
                evaluation_checkpoint.restore(checkpoint_manager.latest_checkpoint).assert_existing_objects_matched()

        self.completed_epochs = int(step_counter)


    @tf.function
    def evaluation_step(self, x_batch_timestep_n_features, x_batch_labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).

        if self.is_knowledge_distill:
            student_logits = self.model(x_batch_timestep_n_features, training=False)
            predictions = tf.nn.softmax(student_logits)
        else:
            predictions = self.model(x_batch_timestep_n_features, training=False)

        validation_loss = self.loss_object(x_batch_labels, predictions)

        # Update metrics
        self.test_loss.update_state(validation_loss)
        self.test_confusion_matrix_metrics.update_state(x_batch_labels, predictions)
        self.test_class_wise_metrics.update_state(x_batch_labels, predictions)
        self.test_average_metrics.update_state(x_batch_labels, predictions)


    def evaluate(self):
        """
            Evaluates a model on test dataset
            Parameters
            ----------

            Returns
            ----------
        """
        eval_label_counter = 0
        counter_print_last_value, counter_print_interval = 0, 100_000

        # Reset metrics
        self.test_loss.reset_states()
        self.test_confusion_matrix_metrics.reset_states()
        self.test_class_wise_metrics.reset_states()
        self.test_average_metrics.reset_states()

        evaluation_logger.info(f"Start evaluation for model '{self.model.name}'.")

        for test_batch_timestep_n_features, test_batch_labels in self.test_dataset:
            eval_label_this_loop_count = test_batch_timestep_n_features.shape[0] * test_batch_timestep_n_features.shape[1]
            if self.is_ensemble:
                test_batch_timestep_n_features = [test_batch_timestep_n_features for _ in range(len(self.model.input))]

            self.evaluation_step(test_batch_timestep_n_features, test_batch_labels)

            eval_label_counter += eval_label_this_loop_count

            if eval_label_counter > (counter_print_last_value + counter_print_interval):
                counter_print_last_value = (eval_label_counter // counter_print_interval) * counter_print_interval # Round down.
                tf.print(f'Test Labels evaluated: {eval_label_counter}')
        
        test_loss_result = self.test_loss.result()
        test_confusion_matrix_results = self.test_confusion_matrix_metrics.result()
        test_class_wise_metric_results = self.test_class_wise_metrics.result()
        test_class_wise_metric_results = convert_ClassWiseMetric_result(test_class_wise_metric_results,
                                                                        self.num_classes)
        test_class_wise_metric_results_string = format_metric_output(test_class_wise_metric_results)
        test_average_metric_results = self.test_average_metrics.result()
        test_average_metric_results = convert_AverageMetric_result(test_average_metric_results)
        test_average_metric_results_string = format_metric_output(test_average_metric_results)

        template = ('After {} epochs:\n'
                    'Test Loss: {:.3f}, \n'
                    'Test Class Wise Metric: {}, \n'
                    'Test Average Metric: {}, \n'
                    '*******************************************')

        evaluation_logger.info(template.format(self.completed_epochs,
                                               test_loss_result,
                                               test_class_wise_metric_results_string,
                                               test_average_metric_results_string))
        evaluation_logger.info(test_confusion_matrix_results)
        evaluation_logger.info("//////////////////////////////////////////////")

        with self.test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss_result, step=self.completed_epochs)
            for name, value in test_class_wise_metric_results.items():
                tf.summary.scalar(name, value, step=self.completed_epochs)

            for name, value in test_average_metric_results.items():
                tf.summary.scalar(name, value, step=self.completed_epochs)

            test_confusion_matrix_fig = format_confusion_matrix_metric_output(self.num_classes,
                                                                              test_confusion_matrix_results)
            tf.summary.image('Evaluation confusion matrix', self._plot_to_image(test_confusion_matrix_fig),
                             step=self.completed_epochs)

        return (test_loss_result, test_average_metric_results['accuracy_macro'], self.completed_epochs)


    def _plot_to_image(self, figure):
        """
            Converts the matplotlib plot specified by 'figure' to a PNG image and
            returns it. The supplied figure is closed and inaccessible after this call.
            Parameters
            ----------
            `figure` (matplotlib.figure) : figure of the plot

            Returns
            ----------
                (tensor):  converted png
        """

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
