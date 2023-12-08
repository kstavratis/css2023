from concurrent.futures.process import ProcessPoolExecutor
from tqdm import tqdm

from src.experiments.experiments import run_comparison_experiments
from src.experiments.utils.prepare_configurations import prepare_experiment_configurations
from src.experiments import nr_experiments, time_steps

def main():

    social_systems_kwargs_list = prepare_experiment_configurations()
    nr_configurations = len(social_systems_kwargs_list)


    # Single threaded run.
    # for kwargs in social_systems_kwargs_list:
    #     print(run_comparison_experiment(kwargs))


    # Although multiple different experiments are run in parallel,
    # the results are not aggregated/collected/stored somewhere
    # where they might be utilized for analysis.
    # TODO IMPLEMENT
    # Data collection/storage.
    executor = ProcessPoolExecutor()
    # Note: callist "list" enforces the evaluation of the map
    # (a.k.a all experiment are conducted before moving to the next code instruction).
    test = list(tqdm(
        executor.map(
            run_comparison_experiments,
            tuple(nr_experiments for _ in range(nr_configurations)),
            tuple(time_steps for _ in range(nr_configurations)),
            social_systems_kwargs_list
        ),
        total=nr_configurations)
    )


    #save all result in csv files
    path = './results/'
    for i, item in enumerate(test):
        conf = social_systems_kwargs_list[i]
        n_agent = conf['graph']
        mu = conf['interaction_intensity']
        theta = conf['opinion_tolerance']
        mean_edge_per_agent = conf['pr_edge_creation']*n_agent
        pr_meeting_friend = conf['pr_friend_default']
        pr_meeting_friend_of_friend = conf['pr_friend_of_friend_default']


        for i in range(nr_experiments):
            backbone_res = item.backbone_opinions_in_experiment(i)
            homophilic_res = item.homophilic_opinions_in_experiment(i)
            config = f'{n_agent}_{mu}_{theta}_{mean_edge_per_agent}'
            homophily_config = f'{pr_meeting_friend}_{pr_meeting_friend_of_friend}'
            backbone_res.to_csv(path+'backbone_'+config+f'_{i}.csv', float_format="%.3f")
            homophilic_res.to_csv(path+'homophilic_'+config+'_'+homophily_config+f'_{i}.csv', float_format="%.3f")
    

if __name__ == '__main__':
    main()