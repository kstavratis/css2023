# Can be changed in case it is decided
# to approach the opinions using probabilities instead.
opinion_value_lb = -1 # lb -> "lower bound"
opinion_value_ub = 1 # ub -> "upper bound"

# These link strengths are only used for initialization
# of the link strengths of the network.
# May be set to -infinity and +infinity instead.
link_strength_lb_init = -10
link_strength_ub_init = 10

# This tolerance value is dependent on the representation
# of the opinion value.
# *DO NOT FORGET TO CHANGE IT ACCORDINGLY!
opinion_tolerance = 0.5

# Interaction intensity.
# You may think of this as the step size
# of a gradient descent algorithm 
interaction_intensity = 0.2

# A value which represents zero, but in actuality isn't.
# It is defined to avoid problematic runtime behaviour
# e.g. division by zero. 
EPSILON = 1e-8
# NOTE: this could be extracted by the numpy library,
# which can extract the per-machine ε.