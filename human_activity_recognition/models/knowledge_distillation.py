import logging

import gin
import tensorflow as tf
import datetime

from models.architectures import load_models

knowledge_distillation_logger = logging.getLogger('knowledge_distillation')

@gin.configurable
class KnowledgeDistill():
    def __init__(self, student_model_desc, dataset_info, run_paths,
                 teacher_model_desc):  # <- configs
        self.student_model_desc = student_model_desc
        self.student_model = None

        self.number_of_classes = dataset_info['number_of_classes']

        self.teacher_model_desc = teacher_model_desc
        self.teacher_model = None

        self.dataset_info = dataset_info
        self.run_paths = run_paths

        self.saved_model_path = run_paths['trained_models_directory']

    def _load_saved_model(self):
        # Model loader
        models = load_models(self.teacher_model_desc, self.dataset_info)

        # Load the first saved model as teacher model
        model_name, loaded_teacher_model = models[0]  # loaded_model: Model which accepts the data from the saved models.
        model_path = self.saved_model_path / loaded_teacher_model.name

        if model_path.exists():
            knowledge_distillation_logger.info(f"Pre-trained Teacher model {loaded_teacher_model.name}")
            knowledge_distillation_logger.info(f"Loading model {loaded_teacher_model.name} from path {str(model_path)}")
            model_path = str(model_path / loaded_teacher_model.name) + '-1'
            # The suffix '-1' is hardcoded, because if models are selected from a 'trained_models' dir. then there should only be one checkpoint.
            # If it is necessary to accept any checkpoint, then the suffix has to be read per model from the files in the dir..

            tf.train.Checkpoint(model=loaded_teacher_model).restore(save_path=model_path)
        else:
            knowledge_distillation_logger.error(
                f"Could not find the path {model_path}. Try training the teacher model first")
            raise TypeError(f"Could not find the path {model_path}. Try training the teacher model first")

        return loaded_teacher_model

    def get_teacher_model(self):
        # load all the saved models
        # list of tuple (model_name, loaded_model)
        try:
            if self.teacher_model is None:
                self.teacher_model = self._load_saved_model()
                # extract the model till the logit layer
                self.teacher_model = tf.keras.Model(inputs=[self.teacher_model.input],
                                                    outputs=[self.teacher_model.get_layer("output_logits").output],
                                                    name=self.teacher_model.name)
                self.teacher_model.summary()
        except Exception as e:
            raise e
        return self.teacher_model

    def get_student_model(self):
        if self.student_model is None:
            _, self.student_model = load_models(self.student_model_desc, self.dataset_info)[0]
            self.student_model = tf.keras.Model(inputs=[self.student_model.input],
                                                outputs=[self.student_model.get_layer("output_logits").output],
                                                name=f"student_{self.student_model.name}")
            self.student_model.summary()
        return self.student_model


