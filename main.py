from tqdm import tqdm

from src.social_system.backbone_social_system import BackboneSocialSystem
from src.social_system.mixins.social_system_with_extremists import SocialSystemWithExtremists

arguments = {
    'nr_agents' : 10_000,
    'positive_extremists_indices' : [0],
    'negative_extremists_indices' : [3]
}

class RunClass(SocialSystemWithExtremists, BackboneSocialSystem):

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