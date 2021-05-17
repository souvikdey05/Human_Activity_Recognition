import logging
import tensorflow as tf
import datetime


ensemble_logger = logging.getLogger('evaluation')

class StackingEnsemble(object):
    def __init__(self, models, checkpoint_path, dataset_info,
                 run_paths):

        self.models = models
        self.number_of_classes = dataset_info['number_of_classes']

        self.level_0_loaded_models = None
        self.ensemble_model = None

        self.checkpoint = checkpoint_path
        self.dataset_info = dataset_info
        self.run_paths = run_paths

        # Summary Writer
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        summary_directory = run_paths['ensemble_log'].parent / current_time
        self.ensemble_summary_writer = tf.summary.create_file_writer(logdir=str(summary_directory / 'ensemble'))

        self.saved_model_path = run_paths['trained_models_directory']

    def _load_saved_models(self):
        loaded_models = list()  # list of tuple (model_name, loaded_model)

        # Load all the saved models
        for model in self.models:
            model_name, loaded_model = model  # loaded_model: Model which accepts the data from the saved models.
            model_path = self.saved_model_path / loaded_model.name

            if model_path.exists():
                ensemble_logger.info(f"Loading model {loaded_model.name} from path {str(model_path)}")
                model_path = str(model_path / loaded_model.name) + '-1'
                # The suffix '-1' is hardcoded, because if models are selected from a 'trained_models' dir. then there should only be one checkpoint.
                # If it is necessary to accept any checkpoint, then the suffix has to be read per model from the files in the dir..

                tf.train.Checkpoint(model=loaded_model).restore(save_path=model_path)
                loaded_models.append((model_name, loaded_model))
            else:
                ensemble_logger.error(
                        f"Could not find the path {model_path}. Try training first")

        return loaded_models

    def _prepare_ensemble_model(self, loaded_models):
        ensemble_inputs = []
        ensemble_outputs = []
        # update all the models to make it non trainable. These models will be used as input heads for the
        # ensemble model
        for i, model in enumerate(loaded_models):
            model_name, model_architecture = model

            for layer in model_architecture.layers:
                # make not trainable
                layer.trainable = False
                # rename to avoid 'unique layer name' issue
                layer._name = 'ensemble_' + str(i + 1) + '_' + layer.name

            ensemble_inputs.append(model_architecture.input)
            ensemble_outputs.append(model_architecture.output)
        dense_units = 100
        dropout_rate = 0.2

        conc_layer = tf.keras.layers.Concatenate(axis=-1)(ensemble_outputs)
        dense_layer = tf.keras.layers.Dense(dense_units, activation='relu')(conc_layer)
        dropout_layer = tf.keras.layers.Dropout(dropout_rate)(dense_layer)
        output = tf.keras.layers.Dense(self.number_of_classes, activation='softmax')(dropout_layer)

        dropout_rate_string = str(dropout_rate).replace('.', '_')
        name = '-'.join(['ensemble_model', str(dense_units), dropout_rate_string])

        ensemble_model = tf.keras.Model(inputs=ensemble_inputs, outputs=output, name=name)
        ensemble_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                               optimizer='adam', metrics=['accuracy'])

        ensemble_model.summary()

        return ensemble_model

    def get_level_0_models(self):
        """
            Returns saved models from trained_models folder for the models mentioned in config main.models
            This models are level 0 base models and are the inputs to the ensemble models

            Parameters
            ----------

            Returns
            -------
                (list): list of tuple (model_name, loaded_model)
        """
        # load all the saved models
        # list of tuple (model_name, loaded_model)
        if self.level_0_loaded_models is None:
            self.level_0_loaded_models = self._load_saved_models()
        return self.level_0_loaded_models

    def get_stacking_ensemble_model(self):
        '''Creates and returns a stacking ensemble model with level 0 models as inputs.

            Parameters
            ----------

            Returns
            -------
                tf.keras.Model
        '''
        # load all the saved models
        loaded_models = self.get_level_0_models()

        if len(loaded_models) < 2:
            ensemble_logger.error("Cannot do ensemble learning with less than 2 models")
            raise ValueError("Cannot do ensemble learning with less than 2 models")

        # prepare the model for training ensemble
        self.ensemble_model = self._prepare_ensemble_model(loaded_models)

        return self.ensemble_model

