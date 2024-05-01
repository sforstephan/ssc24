
parameters = {

    # general simulation parameters
    "time":                         500, # observation period
    "repeat":                       150, # simulation runs
    "n":                            15, # dimensions of the decision problem

    # shocks
    "schock_correlation":           -0.5, # correlation of shocks, select from -1 to +1 

    # decision problem / landscape characteristics
    "matrix":                       "random", # interaction pattern, select from small_blocks, reciprocal,random
    "k":                            7, # k has to be set if matrix is "random", parameter is ignored otherwise

    # decision making and information exchange
    "decision_making_mode":         "centralized_hillclimbing", # select from: decentralized, centralized_hillclimbing
    "information_exchange_mode":    "no", # select from: no, sequential

    # agents
    "incentive_parameter":          0.5,
    "joint_search":                 False,
    "social_search_prob":           0, # if joint search is True, joint search probability needs to be specified
}

