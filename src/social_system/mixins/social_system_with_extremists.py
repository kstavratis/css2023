import numpy as np

from .. import opinion_value_lb, opinion_value_ub
from .. import EPSILON

class SocialSystemWithExtremists:

    def __init__(self, positive_extremists_indices: list, negative_extremists_indices: list, **kwargs) -> None:
        super().__init__(**kwargs)

        self.is_positive_extremist, self.is_negative_extremist =\
            np.zeros_like(self.opinions), np.zeros_like(self.opinions)
        
        self.is_positive_extremist[positive_extremists_indices] = 1
        self.is_negative_extremist[negative_extremists_indices] = 1

        # Transform to boolean masks
        self.is_positive_extremist = self.is_positive_extremist.astype(bool, copy=False)
        self.is_negative_extremist = self.is_negative_extremist.astype(bool, copy=False)

        # Hard set the opinion of the extreme agents to the opinion value boundaries.
        self.opinions[self.is_negative_extremist] =  opinion_value_lb
        self.opinions[self.is_positive_extremist] =  opinion_value_ub


        # Hard set the tolerance of extreme agents to (almost) zero.
        self.tolerances[self.is_positive_extremist] = EPSILON
        self.tolerances[self.is_negative_extremist] = EPSILON

    def _postprocess(self):

        # Reset tolerance of extremists to (almost) zero after the interactions take place.
        self.tolerances[self.is_positive_extremist] = EPSILON
        self.tolerances[self.is_negative_extremist] = EPSILON

        # Reset the opinion of the extreme agents to the opinion value boundaries
        # after the interactions take place
        self.opinions[self.is_negative_extremist] =  opinion_value_lb
        self.opinions[self.is_positive_extremist] =  opinion_value_ub

    

