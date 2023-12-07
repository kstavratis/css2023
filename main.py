from concurrent.futures.process import ProcessPoolExecutor
from tqdm import tqdm

from src.experiments.experiments import run_comparison_experiments
from src.experiments.utils.prepare_configurations import prepare_experiment_configurations
from src.experiments import nr_experiments, time_steps
from src.experiments.loggers.logger import write_social_system_experiment

def main():

    social_systems_kwargs_list = prepare_experiment_configurations()
    nr_configurations = len(social_systems_kwargs_list)
    


    # Single threaded run.
    for kwargs in social_systems_kwargs_list:
        comparison_expms = run_comparison_experiments(20, time_steps, kwargs)
        write_social_system_experiment(comparison_expms.experiment(0))


    # # Although multiple different experiments are run in parallel,
    # # the results are not aggregated/collected/stored somewhere
    # # where they might be utilized for analysis.
    # # TODO IMPLEMENT
    # # Data collection/storage.
    # executor = ProcessPoolExecutor()
    # # Note: callist "list" enforces the evaluation of the map
    # # (a.k.a all experiment are conducted before moving to the next code instruction).
    # comparison_experiments_test = list(tqdm(
    #     executor.map(
    #         run_comparison_experiments,
    #         tuple(nr_experiments for _ in range(nr_configurations)),
    #         tuple(time_steps for _ in range(nr_configurations)),
    #         social_systems_kwargs_list
    #     ),
    #     total=nr_configurations)
    # )
    

if __name__ == '__main__':
    main()