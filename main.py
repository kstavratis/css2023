from concurrent.futures.process import ProcessPoolExecutor

from tqdm import tqdm

from src.social_system.utils.experiments import run_comparison_experiment



def main():

    # Arguments declaration and initialization
    nr_experiments = 5
    nr_agents = 100
    time_steps = 1#_000

    kwargs_list = []

    # μ = θ = 0.5 := "Centrality"
    # μ = θ = 0.25 := "Debility"
    # μ = 0.25, θ = 0.5 := "Tolerability"
    # μ = 0.5, θ = 0.25 := "Susceptibility"
    for mu, theta in ((0.5, 0.5), (0.25, 0.25), (0.25, 0.5), (0.5, 0.25)):
        for expected_edges_per_agent in (7, 11, 20):
            for pr_friend, pr_friend_of_friend in ((0.8, 0.1), (0.9, 0.09), (0.95, 0.04)):
                kwargs = {
                    'time_steps': time_steps,

                    'graph' : nr_agents,
                    'interaction_intensity': mu,
                    'opinion_tolerance': theta,

                    'pr_edge_creation': expected_edges_per_agent / nr_agents,
                    'pr_positive': 0.0, # In the current implementation, this parameter is ignored.
                    'pr_friend': pr_friend,
                    'pr_friend_of_friend': pr_friend_of_friend
                }
                kwargs_list.append(kwargs)



    # Single threaded run.
    # for kwargs in kwargs_list:
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
        executor.map(run_comparison_experiment, kwargs_list),
        total=4*3*3)
    )

    for item in test:
        print(item)
    

if __name__ == '__main__':
    main()