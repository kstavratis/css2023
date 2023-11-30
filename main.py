from tqdm import tqdm

from src.social_system.backbone_social_system import BackboneSocialSystem
from src.social_system.homophilic_social_system import HomophilicSocialSystem
from src.social_system.mixins.social_system_with_extremists import SocialSystemWithExtremists

nr_agents = 100
expected_edges_per_agent = 2

arguments = {
    'nr_agents' : nr_agents,
    'pr_edge_creation' : expected_edges_per_agent/nr_agents, # The idea is for the probability to be `1/nr_agents`.
    'pr_positive': 0.0, # In the current implementation, this parameter is ignored.
    'positive_extremists_indices' : [0],
    'negative_extremists_indices' : [3]
}

class RunClass(HomophilicSocialSystem):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

def main():
    
    test = RunClass(**arguments)
    print(f'Initial opinions:\n{test.opinions}')
    print(f'Initial tolerances:\n{test.tolerances}') 
    for _ in tqdm(range(1000)):
        test.step()

    print(f'Final opinions:\n{test.opinions}')
    print(f'Final tolerances:\n{test.tolerances}')

if __name__ == '__main__':
    main()