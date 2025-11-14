import os
import logging

from utils.utils import set_global_seed

seed = 42
set_global_seed(seed)

class BaseConfig:
    project_directory = os.path.dirname(os.path.realpath(__file__))

class DatasetConfig:
    dataset_directory = os.path.join(BaseConfig.project_directory, "datasets")

class ModelConfig:
    
    model_base_directory = os.path.join(BaseConfig.project_directory, "experiments")

    model_plot_directory = os.path.join(model_base_directory, "plots")
    ood_evaluation_directory = os.path.join(model_base_directory, "results")
    model_accuracy_directory = os.path.join(model_base_directory, "accuracy")
    runtime_log_directory = os.path.join(model_base_directory, "runtime_log")
    model_statistics_directory = os.path.join(model_base_directory, "statistics")
    model_checkpoint_directory = os.path.join(model_base_directory, "checkpoints")
    model_percentile_directory = os.path.join(model_base_directory, "percentile")
    tensorboard_log_directory = os.path.join(model_base_directory, "tensorboard_log")
    

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(module)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')


if __name__ == '__main__':
    print(BaseConfig.project_directory)
    print(DatasetConfig.dataset_directory)
    print(ModelConfig.model_base_directory)
    print(ModelConfig.model_checkpoint_directory)
    print(ModelConfig.runtime_log_directory)
    print(ModelConfig.tensorboard_log_directory)