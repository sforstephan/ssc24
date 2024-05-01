from NK import *
from matrices import *
from parameters import parameters as simulation_parameters
import numpy as np
import scipy.stats as st
import json
import sys
import copy
from datetime import datetime


def main():

    parameters = add_fixed_parameters(simulation_parameters)

    # get dependencymap
    if parameters["matrix"] != "random" and parameters["matrix"] in matrices.keys():
        interaction_pattern = matrices[parameters["matrix"]]
    elif parameters["matrix"] == "random":
        interaction_pattern = get_random_matrix(parameters["n"], parameters["k"])
    else:
        raise ValueError("Invalid parameter for matrix")

    scenarios = 1

    for scen in range(scenarios):

        # initialize datastructure to save the results
        results = dict()
        results["fitness"] = dict()
        results["parameters"] = dict()
        results["parameters"] = copy.deepcopy(parameters)
        results["time"] = dict()
        results["time"]["start"] = datetime.now()

        for wdh in range(parameters["repeat"]):

            # datastructure for results
            results["fitness"][wdh] = dict()

            # in the inital round of whenever a new landscape should be generated
            if wdh == 0 or wdh % parameters["new_landscape_after"] == 0:
                # if the interaction pattern is random, create a new pattern
                if parameters["matrix"] == "random":
                    interaction_pattern = get_random_matrix(
                        parameters["n"], parameters["k"]
                    )
                # initialize the landscape
                try:
                    landscape
                except NameError:
                    var_exists = False
                else:
                    var_exists = True

                if var_exists:
                    del landscape

                landscape = landscape = Landscape(interaction_pattern)

            # initialize coordinator object
            # coordinator is re-created in every repetition of the simulation
            # first check if coordinator exist, if so, delete
            try:
                coordinator
            except NameError:
                var_exists = False
            else:
                var_exists = True

            if var_exists:
                del coordinator

            coordinator = Coordinator(
                landscape.get_dependency_map(),
                parameters["number_of_agents"],
                parameters["decision_making_mode"],
                parameters["information_exchange_mode"],
                parameters["search_mode"],
                parameters["sequential_allocation"],
                incentive_parameter=parameters["incentive_parameter"],
                number_of_proposals=parameters["number_of_proposals"],
            )

            # loop over all agents to initalize agent objects
            # agents are re-created in every repetition of the simulation
            # first check if agents exist, if so, delete
            try:
                agents
            except NameError:
                var_exists = False
            else:
                var_exists = True

            if var_exists:
                del agents

            agents = list()
            for i in range(parameters["number_of_agents"]):
                # initialize manager agents
                agents.append(
                    Agent(
                        i,
                        landscape.get_dependency_map(),
                        parameters["number_of_agents"],
                        parameters["decision_making_mode"],
                        parameters["information_exchange_mode"],
                        parameters["search_mode"],
                        parameters["min_genes"],
                        parameters["max_genes"],
                        error_mean=parameters["error_mean"],
                        error_std=parameters["error_std"],
                        incentive_parameter=parameters["incentive_parameter"],
                        number_of_proposals=parameters["number_of_proposals"],
                    )
                )
                # update initial task allocation for every agent
                agents[i].update_responsibility(coordinator.get_allocation()[i])
                # update initial position for every agent
                agents[i].update_position(coordinator.get_position(), initial=True)

            if wdh == 0:
                # initialize network class
                # first check if network exist, if so, delete
                try:
                    network
                except NameError:
                    var_exists = False
                else:
                    var_exists = True

                if var_exists:
                    del network

                network = Network(
                    landscape.get_dependency_map(),
                    parameters["number_of_agents"],
                    social_learning_probability=parameters["social_learning_prob"],
                    social_search_probability=parameters["social_search_prob"],
                )

            # loop over timesteps
            for timestep in range(parameters["time"]):

                # if initial period, just store results
                if timestep == 0:

                    results["fitness"][wdh][timestep] = coordinator.compute_utility(
                        landscape, coordinator.get_position()
                    )

                # if it is not the initial period
                else:
                    # correlated schock after 250 periods
                    if timestep == 250:
                        landscape.recompute_contributions(
                            parameters["schock_correlation"]
                        )

                    # if agents only search individually
                    if parameters["joint_search"] == False:

                        # if stochastic hillclimbing
                        if parameters["search_mode"] == "hillclimbing":
                            # loop over all agents
                            for i in range(len(agents)):
                                # if sequential information exchange in period and the decision-making agent is not the first one (index zero)
                                if (
                                    parameters["information_exchange_mode"]
                                    == "sequential"
                                    and i > 0
                                ):
                                    # agent i gets information on the proposals of all agents that made the decision befor it
                                    # this means, agents 0 to i-1 inform agent i about the decisions that the agents made
                                    for j in range(0, i):
                                        agents[i].update_information_state(
                                            coordinator.get_proposals_agent(j)
                                        )

                                # compute proposals and submit to coordinator
                                proposal = agents[i].hillclimbing(
                                    landscape, parameters["number_of_discoveries"]
                                )
                                coordinator.update_proposals(i, proposal)
                                # print(f"proposal agent {i} is {proposal}")

                        else:
                            sys.exit("Search mode not specified!")

                    elif parameters["joint_search"] == True:
                        # get social connections for joint search in the current counr
                        connections = network.get_social_connections(search=True)

                        # get list of agents to loop over
                        agent_loop = set(connections.keys())
                        # get set of agents that still need to perform a search
                        tmp_agents = copy.deepcopy(agent_loop)

                        # loop over agents
                        for agent in agent_loop:

                            # if the agent still has to perform the search
                            if agent in tmp_agents:

                                # remove that agent from the set
                                tmp_agents.remove(agent)

                                # if that agent is selected to perform a single search
                                if connections[agent]["joint_search"] == False:

                                    # if the search method is hillclimbing
                                    if parameters["search_mode"] == "hillclimbing":
                                        # let the agent perform hillclimbing
                                        # compute proposals and submit to coordinator
                                        proposal = agents[agent].hillclimbing(
                                            landscape,
                                            parameters["number_of_discoveries"],
                                        )
                                        coordinator.update_proposals(agent, proposal)
                                    else:
                                        sys.exit("1. Search mode not specified!")

                                # if the agent is selected to perform a joint search
                                else:
                                    # get the agent's search partner
                                    # and remoe it from the list of agents (to make sure every agent performs search only once)
                                    partner = connections[agent]["partner"]
                                    tmp_agents.remove(partner)

                                    # let the agent and the search partner discover alternative configurations for their partial bitstrings
                                    alternatives_agent = agents[
                                        agent
                                    ].get_alternative_positions(
                                        landscape, parameters["number_of_discoveries"]
                                    )
                                    alternatives_partner = agents[
                                        partner
                                    ].get_alternative_positions(
                                        landscape, parameters["number_of_discoveries"]
                                    )

                                    # inform agent about the other agent's discoveries and let them compute their utilities
                                    alternatives_utility_agent = agents[
                                        agent
                                    ].get_utility_social_search(
                                        landscape,
                                        alternatives_agent,
                                        alternatives_partner,
                                        parameters["number_of_discoveries"],
                                    )
                                    alternatives_utility_partner = agents[
                                        partner
                                    ].get_utility_social_search(
                                        landscape,
                                        alternatives_partner,
                                        alternatives_agent,
                                        parameters["number_of_discoveries"],
                                    )

                                    # if the search method is hillclimbing
                                    if parameters["search_mode"] == "hillclimbing":
                                        proposals_agent, proposals_partner = agents[
                                            agent
                                        ].social_hillclimbing(
                                            alternatives_utility_agent,
                                            alternatives_utility_partner,
                                            parameters["number_of_discoveries"],
                                        )
                                    else:
                                        sys.exit("2. Search mode not specified!")

                                    # forward the proposals to the coordinator
                                    coordinator.update_proposals(agent, proposals_agent)
                                    coordinator.update_proposals(
                                        partner, proposals_partner
                                    )

                    else:
                        sys.exit("3. Search mode not specified!")

                    # coordinator's decision
                    coordinator.make_decision(
                        landscape,
                        agents,
                        coordinator.get_decision_making_mode(),
                        initial_temperature=parameters["initial_temperature"],
                        alpha=parameters["alpha"],
                        time=timestep,
                    )

                    results["fitness"][wdh][timestep] = coordinator.compute_utility(
                        landscape, coordinator.get_position()
                    )

        results["time"]["end"] = datetime.now()
        results["time"]["elapsed"] = str(
            results["time"]["end"] - results["time"]["start"]
        )
        results["time"]["start"] = results["time"]["start"].isoformat()
        results["time"]["end"] = results["time"]["end"].isoformat()

        filename = "results.json"

        with open(filename, "w") as file:
            json.dump(results, file)

        print(f"Simluation finished, elapsed time: -> {results['time']['elapsed']}")


def add_fixed_parameters(parameters):
    # params = deepcopy(parameters)
    fixed_parameters = {
        "number_of_proposals": 2,
        "number_of_agents": 5,
        "number_of_discoveries": 1,
        "min_genes": 1,
        "max_genes": 7,
        "error_mean": 0.0,
        "error_std": 0.00,
        "sequential_allocation": True,
        "search_mode": "hillclimbing",
        "realloc_after": 10,
        "new_landscape_after": 25,
        "initial_temperature": 0.8,
        "alpha": 0.9,
        "learning_mode": "own",
        "social_learning_prob": 0,  # float
        "bottom_up_allocation": False,
        "swapping_mode": "none",
        "alloc_error_mean": 0,
        "alloc_error_std": 0.01,
        "alloc_weight": 1,
    }

    parameters.update(fixed_parameters)

    return parameters


if __name__ == "__main__":
    main()
