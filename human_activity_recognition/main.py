import gin
import logging
import absl
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.python.util.deprecation import _PRINT_DEPRECATION_WARNINGS
import pathlib
import shutil

from tune import hyperparameter_tuning
from train import Trainer, KDTrainer
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import load_models
from evaluation.evaluation import Evaluator
from evaluation.ensemble import StackingEnsemble
from models.knowledge_distillation import KnowledgeDistill

FLAGS = absl.flags.FLAGS
# Use --train or --train=true to set this flag, or nothing, true is default.
# Use --notrain or --train=false to set this flag to false.
absl.flags.DEFINE_boolean(name='train', default=True,  help='Specify whether to train a model.')
absl.flags.DEFINE_boolean(name='eval',  default=True,  help='Specify whether to evaluate a model.')
absl.flags.DEFINE_boolean(name='ensem', default=False, help='Specify whether to use ensemble learning.')

# Configure the number of threads used by tensorflow
# tf.config.threading.set_intra_op_parallelism_threads(3)
# tf.config.threading.set_inter_op_parallelism_threads(3)



def setup_checkpoint_loading(model_name, resume_checkpoint, resume_model_path, run_paths):
    if not resume_model_path:  # resume_checkpoint is the path of a checkpoint file
        return resume_checkpoint

    else:  # resume_checkpoint is a checkpoint prefix
        summaries_directory = pathlib.Path(resume_model_path) / 'summaries'
        # Copy the summaries of the previous attempt
        shutil.copytree(src=str(summaries_directory), dst=run_paths['model_directories'][model_name]['summaries'],
                        dirs_exist_ok=True)

        if not resume_checkpoint:  # Load latest checkpoint
            return str(pathlib.Path(resume_model_path) / 'checkpoints')
        else:  # Load specified checkpoint
            return str(pathlib.Path(resume_model_path) / 'checkpoints' / resume_checkpoint)


def verify_configs (models) -> None:
    # Verify the models type
    if isinstance(models, dict) and ('model' not in models):
       raise ValueError("The model dictionary should contain the key 'model' with a name.")
    
    elif isinstance(models, list):
        for model in models:
            if isinstance(model, dict) and ('model' not in model):
                raise ValueError("The model dictionary should contain the key 'model' with name.")
    
    else:
        raise ValueError("The model should be a dictionary or list of dictonaries.")

    # Verify configs
    if FLAGS.ensem and (not isinstance(models, list) or len(models) < 2):
        raise ValueError("For Ensemble Learning, train more than one model.")


def ensemble_learning (models, train_dataset, validation_dataset, test_dataset, dataset_info,
                        run_paths, last_checkpoint, resume_checkpoint, resume_model_path) -> None:
    
    ensemble = StackingEnsemble(models, None, dataset_info['number_of_classes'],
                                        validation_dataset, test_dataset, dataset_info, run_paths)
    level_0_loaded_models = ensemble.get_level_0_models()  # list of tuple (model_name, loaded_model)

    # for all the loaded models do an evaluation to see how it performs individually
    for model in level_0_loaded_models:
        model_name, model_architecture = model
        is_ensemble_evaluation = False
        is_knowledge_distill = False

        # restore_from_checkpoint is False here because this models are already loaded from
        # trained_models folder. So no need to restore it again
        _, _, _ = Evaluator(model_architecture, is_ensemble_evaluation, is_knowledge_distill,
                            last_checkpoint, test_dataset, dataset_info, run_paths,
                            restore_from_checkpoint=False).evaluate()

    ensemble_model = ensemble.get_stacking_ensemble_model()
    utils_params.generate_model_directories(ensemble_model.name, run_paths)

    resume_checkpoint_path = setup_checkpoint_loading(ensemble_model.name,
                                                        resume_checkpoint, resume_model_path, run_paths)

    is_ensemble_training = True
    ensemble_trainer = Trainer(ensemble_model, is_ensemble_training, train_dataset,
                                validation_dataset, dataset_info, run_paths, resume_checkpoint_path)
    last_checkpoint = ensemble_trainer.train()

    is_ensemble_evaluation = True
    is_knowledge_distill = False
    _, _, _ = Evaluator(ensemble_model, is_ensemble_evaluation, is_knowledge_distill,
                        last_checkpoint, test_dataset, dataset_info, run_paths).evaluate()


@gin.configurable
def main(argv,
         models, use_hyperparameter_tuning, use_knowledge_distillation, # <- configs
         resume_checkpoint, resume_model_path, evaluation_checkpoint): # <- configs

    verify_configs(models)
    
    # Generate folder structures
    run_paths = utils_params.generate_run_directory()
    # Set loggers
    utils_misc.set_loggers(paths=run_paths, logging_level=logging.INFO)
    # Save gin config
    utils_params.save_config(run_paths['path_gin'], gin.config_str() )

    # Setup dataset pipeline
    train_dataset, validation_dataset, test_dataset, dataset_info = datasets.load()
    
    # Features=(32, 250, 6) Label=(32, 250, 1)
    # for d in train_dataset.take(1):
    #     print(d)

    # models = load_models(models, dataset_info)
    #
    # for index, model in enumerate(models):
    #     model_name, model_architecture = model
    #     print(model_name)
    #     model_architecture.summary()
    #     print("------------------")

    if use_hyperparameter_tuning:
        logging.info("Hyper-parameter tuning set to True. Starting Hyper-parameter tuning...")
        # # delete the previous run directories inside the main 'hyperparameter_tuning' directory
        # utils_params.delete_previous_hyperparameter_runs(run_paths)

        # Now start the runs for all the hyperparameters.
        # All the run specific checkpoints and summaries will be saved under 'run_<datetime>'
        # under 'experiment' folder.
        hyperparameter_tuning(models, run_paths)

    elif FLAGS.train or FLAGS.eval or FLAGS.ensem:
        if not use_knowledge_distillation:
            # Model loader
            models = load_models(models, dataset_info)

            last_checkpoint = ''

            for index, model in enumerate(models):
                model_name, model_architecture = model
                model_architecture.summary()
                utils_params.generate_model_directories(model_architecture.name, run_paths)

                resume_checkpoint_path = ''
                # Load checkpoint if this is the first model.
                if (index == 0) and (resume_checkpoint or resume_model_path):
                    resume_checkpoint_path = setup_checkpoint_loading(model_architecture.name,
                                                                      resume_checkpoint, resume_model_path, run_paths)

                if FLAGS.train:
                    is_ensemble_training = False
                    trainer = Trainer(model_architecture, is_ensemble_training, train_dataset, validation_dataset,
                                      dataset_info, run_paths, resume_checkpoint_path)
                    last_checkpoint = trainer.train()

                if FLAGS.eval and not FLAGS.ensem:
                    # no need to evaluate individual models here for FLAGS.ensem=True,
                    # because if eFLAGS.ensem=True then it will be evaluated in the next
                    # part of the code below for models individually as well as for
                    # ensemble model
                    if not FLAGS.train:  # Evaluate a saved model
                        last_checkpoint = evaluation_checkpoint

                    is_ensemble_evaluation = False
                    is_knowledge_distill = False
                    _, _, _ = Evaluator(model_architecture, is_ensemble_evaluation, is_knowledge_distill,
                                        last_checkpoint, test_dataset, dataset_info, run_paths).evaluate()

            if FLAGS.ensem:
                ensemble_learning (models, train_dataset, validation_dataset, test_dataset, dataset_info,
                                   run_paths, last_checkpoint, resume_checkpoint, resume_model_path)

        else:
            logging.info("Knowledge Distillation set to True. Starting knowledge distillation...")

            kd_obj = KnowledgeDistill(models, dataset_info, run_paths)
            teacher_model = kd_obj.get_teacher_model()
            student_model = kd_obj.get_student_model()

            utils_params.generate_model_directories(student_model.name, run_paths)

            resume_checkpoint_path = ''
            # Load checkpoint if this is the first model.
            if resume_checkpoint or resume_model_path:
                resume_checkpoint_path = setup_checkpoint_loading(student_model.name,
                                                                  resume_checkpoint, resume_model_path, run_paths)

            if FLAGS.train:
                trainer = KDTrainer(teacher_model, student_model, train_dataset, validation_dataset,
                                  dataset_info, run_paths, resume_checkpoint_path)
                last_checkpoint = trainer.train()

            if FLAGS.eval:
                if not FLAGS.train:  # Evaluate a saved model
                    last_checkpoint = evaluation_checkpoint

                is_ensemble_evaluation = False
                is_knowledge_distill = True
                _, _, _ = Evaluator(student_model, is_ensemble_evaluation, is_knowledge_distill,
                                    last_checkpoint, test_dataset, dataset_info, run_paths).evaluate()
            



if __name__ == '__main__':
    gin_config_path = pathlib.Path(__file__).parent / 'configs' / 'config.gin'
    gin.parse_config_files_and_bindings([gin_config_path], [])
    absl.app.run(main)
