from pathlib import Path

import pandas as pd

# ==================== Initialize path variables START ====================

# cfp := "current file path"
# Using `resolve` as per the official documentation's recommendation
# https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.parent
__cfp = Path(__file__).resolve()
# Q: Why this weird syntax?
# A: This syntax raise an exception in case that the RHS does not contain
#   only *one* element
[__root_dir] = [parent_path.parent for parent_path in __cfp.parents if parent_path.name == 'src']

# ==================== Initialize path variables FINISH ====================

def write_social_system_experiment(experiment: pd.DataFrame):

    backbone_data, homophilic_data = experiment

    path_to_store = Path(__root_dir / 'experimental_results')
    path_to_store.mkdir(exist_ok=True)

    backbone_data.to_csv(path_to_store / 'BackboneTest.csv')
    homophilic_data.to_csv(path_to_store / 'HomophilicTest.csv')
    


    # experiment.to_csv()