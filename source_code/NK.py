import numpy as np
import math
import random as rnd
import copy
from scipy.stats import beta
import sys


class NK:
    def __init__(self, dependencymap: np.array):
        self.n = self.get_n(dependencymap)
        self.k = self.get_k(dependencymap)
        self.dependencymap = dependencymap

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, var: int):
        self._n = var

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, var: int):
        self._k = var

    @property
    def dependencymap(self):
        return self._dependencymap

    @dependencymap.setter
    def dependencymap(self, dependencymap: np.ndarray):
        self._dependencymap = dependencymap

    @staticmethod
    def get_n(dependencymap: np.array) -> int:
        """
        Returns the number of genes in the genome
        :param dependencymap: interdependence matrix
        :return: number of genes in genome
        """
        return dependencymap.shape[0]

    @staticmethod
    def get_k(dependencymap: np.array) -> int:
        """
        get_k(dependencymap: np.array) -> int

        Returns the number of interdependencies based on the dependency matrix.

        :param dependencymap: (np.array): The interdependence matrix.

        :return int: The number of interdependencies per gene.
        """
        return dependencymap[0].sum() - 1

    def convert_number_to_bin_nparray(self, var: int) -> np.array:
        """
        convert_number_to_bin_nparray(self, var: int) -> np.array

        Converts a number to a binary bitstring and returns the bitstring in a numpy array.

        Parameters:
        var (int): The number to be converted to a binary bitstring.

        Returns:
        np.array: The binary bitstring representation of var as a numpy array.
        """
        # convert int into binary bitstring
        rd: str = bin(var)[2:].zfill(self.n)
        # unpack, convert to int and into numpy array
        rd_unpacked: list = [*rd]
        for i in range(len(rd_unpacked)):
            rd_unpacked[i] = int(rd_unpacked[i])
        return np.array(rd_unpacked)

    def get_dependency_map(self) -> np.array:
        """
        get_dependency_map(self) -> np.array

        Returns the dependency map attribute of the object.

        Parameters:
        None

        Returns:
        np.array: The dependency map attribute of the object.
        """
        return self.dependencymap


class Network(NK):
    def __init__(
        self,
        dependencymap: np.array,
        agents: int,
        social_learning_probability: float = 0.0,
        social_search_probability: float = 0.5,
    ):
        super().__init__(dependencymap)
        self.agents = agents
        self.social_learning_probability = social_learning_probability
        self.social_search_probability = social_search_probability

    @property
    def agents(self):
        return self._agents

    @agents.setter
    def agents(self, var: int):
        self._agents = var

    @property
    def social_learning_probability(self):
        return self._social_learning_probability

    @social_learning_probability.setter
    def social_learning_probability(self, var: float):
        self._social_learning_probability = var

    @property
    def social_search_probability(self):
        return self._social_search_probability

    @social_search_probability.setter
    def social_search_probability(self, var: float):
        self._social_search_probability = var

    @classmethod
    def get(
        cls,
        dependencymap: np.ndarray,
        agents: int,
        social_learning_probability: float,
        social_search_probability: float,
    ) -> object:
        return cls(
            dependencymap,
            agents,
            social_learning_probability,
            social_search_probability,
        )

    def get_social_connections(
        self, information_sharing: bool = False, search: bool = False
    ) -> dict:
        """
        get_social_connections(self, information_sharing: bool = False, search: bool = False) -> dict:

        Computes whether agents share their learned information or perform a joint search. The function returns a dictionary of connections between agents, where the keys are the agent indices and the values are dictionaries with keys "sharing" or "joint_search" indicating whether the agent is participating in information sharing or a joint search, respectively, and "partner" indicating the index of the agent's partner.

        Parameters:
        information_sharing (bool, optional): Set to True if the function should compute information sharing connections. Defaults to False.
        search (bool, optional): Set to True if the function should compute joint search connections. Defaults to False.

        Returns:
        dict: A dictionary of connections between agents.
        """
        connections = dict()
        available_agents = set(range(self.agents))

        for i in range(self.agents):
            # print(f"available agents are {available_agents}")
            # if there are agents left to share information
            if i in available_agents:
                if len(available_agents) > 1:
                    prob = np.random.uniform(0, 1)

                    if information_sharing == True:
                        if prob < self.social_learning_probability:
                            available_agents.remove(i)
                            peer_agent = np.random.choice(list(available_agents))
                            available_agents.remove(peer_agent)
                            connections[i] = dict()
                            connections[i]["sharing"] = True
                            connections[i]["partner"] = peer_agent
                            connections[peer_agent] = dict()
                            connections[peer_agent]["sharing"] = True
                            connections[peer_agent]["partner"] = i
                        else:
                            connections[i] = dict()
                            connections[i]["sharing"] = False
                            available_agents.remove(i)

                    elif search == True:
                        if prob < self.social_search_probability:
                            available_agents.remove(i)
                            peer_agent = np.random.choice(list(available_agents))
                            available_agents.remove(peer_agent)
                            connections[i] = dict()
                            connections[i]["joint_search"] = True
                            connections[i]["partner"] = peer_agent
                            connections[peer_agent] = dict()
                            connections[peer_agent]["joint_search"] = True
                            connections[peer_agent]["partner"] = i
                        else:
                            connections[i] = dict()
                            connections[i]["joint_search"] = False
                            available_agents.remove(i)

                    else:
                        sys.exit("Type of social connection not specified")

                else:
                    connections[i] = dict()
                    if information_sharing == True:
                        connections[i]["sharing"] = False
                    elif search == True:
                        connections[i]["joint_search"] = False
                    else:
                        sys.exit("Type of social connection not specified")

            else:
                continue

        return connections


class Person(NK):
    def __init__(
        self,
        dependencymap: np.array,
        agents: int,
        decision_making_mode: str,
        information_exchange_mode: str,
        search_mode: str,
        error_mean: float,
        error_std: float,
        incentive_parameter: float,
        number_of_proposals: int,
    ):
        super().__init__(dependencymap)
        self.error_mean = error_mean
        self.error_std = error_std
        self.incentive_parameter = incentive_parameter
        self.agents = agents
        self.position = np.empty(self.n)
        self.decision_making_mode = decision_making_mode
        self.information_exchange_mode = information_exchange_mode
        self.search_mode = search_mode
        if search_mode == "simulated_annealing" and number_of_proposals != 2:
            self.number_of_proposals = 2
            print(
                f"For simulated annealing, the numbber of proposals must be equal to two. Number of proposals is set to two."
            )
        else:
            self.number_of_proposals = number_of_proposals

    @property
    def error_mean(self):
        return self._error_mean

    @error_mean.setter
    def error_mean(self, var: float):
        self._error_mean = var

    @property
    def error_std(self):
        return self._error_std

    @error_std.setter
    def error_std(self, var: float):
        self._error_std = var

    @property
    def incentive_parameter(self):
        return self._incentive_parameter

    @incentive_parameter.setter
    def incentive_parameter(self, var: float):
        self._incentive_parameter = var

    @property
    def agents(self):
        return self._agents

    @agents.setter
    def agents(self, var: int):
        self._agents = var

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, var: np.ndarray):
        self._position = var

    @property
    def decision_making_mode(self):
        return self._decision_making_mode

    @decision_making_mode.setter
    def decision_making_mode(self, var: str):
        self._decision_making_mode = var

    @property
    def information_exchange_mode(self):
        return self._information_exchange_mode

    @information_exchange_mode.setter
    def information_exchange_mode(self, var: str):
        self._information_exchange_mode = var

    @property
    def number_of_proposals(self):
        return self._number_of_proposals

    @number_of_proposals.setter
    def number_of_proposals(self, var: int):
        self._number_of_proposals = var

    @property
    def search_mode(self):
        return self._search_mode

    @search_mode.setter
    def search_mode(self, var: str):
        self._search_mode = var

    @classmethod
    def get(
        cls,
        dependencymap: np.ndarray,
        agents: int,
        decision_making_mode: str,
        information_exchange_mode: str,
        search_mode: str,
        error_mean: float,
        error_std: float,
        incentive_parameter: float,
        number_of_proposals: int,
    ) -> object:
        return cls(
            dependencymap,
            agents,
            decision_making_mode,
            information_exchange_mode,
            search_mode,
            error_mean,
            error_std,
            incentive_parameter,
            number_of_proposals,
        )

    def get_position(self) -> np.array:
        """
        get_position(self) -> np.array

        Returns the position attribute of the object.

        Parameters:
        None

        Returns:
        np.array: The position attribute of the object.
        """
        return self.position


class Landscape(NK):
    def __init__(self, dependencymap: np.array, alleles: int = 2):
        if self.check_dependencymap(dependencymap):
            super().__init__(dependencymap)
            # Options per gene, default setting is 2, i.e. binary problem
            self.alleles = alleles
            # random fitness contributions
            self.contributions = np.random.uniform(
                0, 1, self.get_required_fitness_contributions()
            )
            # lookup table to compute indices for fitness contributions
            self.lookuptable = np.zeros([self.n, self.k + 1])
            for i in range(self.n):
                k = 0
                for j in range(self.n):
                    if self.dependencymap[i, j] == 1:
                        self._lookuptable[i, k] = j
                        k = k + 1
            # store the position of the global maximum in the landscape
            self.global_max_position, self.global_max_fitness = self.get_global_max()

        else:
            raise ValueError("Invalid interaction matrix")

    def __str__(self) -> str:
        return (
            f"NK landscape with N = {self.n} Genes and K = {self.k} interdependencies."
        )

    @property
    def global_max_position(self):
        return self._global_max_position

    @global_max_position.setter
    def global_max_position(self, var: np.ndarray):
        self._global_max_position = var

    @property
    def contributions(self):
        return self._contributions

    @contributions.setter
    def contributions(self, var: np.array):
        self._contributions = var

    @property
    def lookuptable(self):
        return self._lookuptable

    @lookuptable.setter
    def lookuptable(self, var: np.ndarray):
        self._lookuptable = var

    @property
    def alleles(self):
        return self._alleles

    @alleles.setter
    def alleles(self, var: int):
        self._alleles = var

    @classmethod
    def get(cls, dependencymap: np.ndarray, alleles: int = 2) -> object:
        """
        Initializes the NK landscape
        :param dependencymap: square interaction matrix of datatype boolean
        :param alleles: options per gene, optimal parameter, set to two if no input (results in binary options for every gene)
        :return: NK landscape object
        """
        return cls(dependencymap, alleles)

    @staticmethod
    def check_dependencymap(dependencymap: np.ndarray) -> bool:
        """
        check_dependencymap(dependencymap: np.ndarray) -> bool

        Checks the interaction matrix for correctness. The method checks for:
        1. Two dimensions
        2. Square matrix
        3. Ones along the main diagonal
        4. Symmetric patterns, i.e., if the sum in all columns and all rows is the same

        Parameters:
        dependencymap (np.ndarray): The interaction pattern in the form of a numpy array.

        Returns:
        bool: True if all conditions are met, False otherwise.
        """
        # checks
        is_two_dimensional: bool = len(dependencymap.shape) == 2
        is_square: bool = dependencymap.shape[0] == dependencymap.shape[1]
        is_one_along_diagonal: bool = all(
            values == 1 for values in np.diagonal(dependencymap)
        )
        # check if sum of columns and rows in the array are all the same
        is_symmetric: bool = all(
            dependencymap[i, :].sum() == dependencymap[0, :].sum()
            and dependencymap[:, i].sum() == dependencymap[0, :].sum()
            for i in range(dependencymap.shape[0])
        )
        return (
            is_two_dimensional and is_square and is_one_along_diagonal and is_symmetric
        )

    def get_required_fitness_contributions(self) -> int:
        """
        get_required_fitness_contributions(self) -> int

        Returns the number of fitness contributions required to compute the NK landscape.

        Parameters:
        None

        Returns:
        int: The number of required fitness contributions.
        """
        return self.n * self.alleles ** (self.k + 1)

    def check_genome(self, genome: np.ndarray) -> None:
        """
        check_genome(self, genome: np.ndarray) -> None

        Checks whether a genome is valid, i.e., of the right length (n) and has only binary values.

        Parameters:
        genome (np.ndarray): A numpy array containing 0 and 1 of length n.

        Returns:
        None

        Raises:
        ValueError: If genome is invalid.
        """
        is_length_n = len(genome) == self.n
        is_binary = all(values == 1 or values == 0 for values in genome)
        if is_length_n and is_binary:
            pass
        else:
            raise ValueError("Invalid genome")

    def get_fitness_gene(
        self, idx: int, genome: np.ndarray, normalized: bool = True
    ) -> float:
        """
        get_fitness_gene(self, idx: int, genome: np.ndarray, normalized: bool = True) -> float

        Returns the contribution of one gene to the fitness of the genome.

        Parameters:
        idx (int): The index of the gene for which the fitness contribution should be returned.
        genome (np.ndarray): The configuration of the genome.
        normalized (bool, optional): Indicates whether the absolute or the normalized fitness is returned by the function. Defaults to True.

        Returns:
        float: The fitness contribution.
        """
        self.check_genome(genome)
        tmp_idx: int = 0
        for i in range(self.k + 1):
            if genome[int(self.lookuptable[idx, i])] == 1:
                tmp_idx = 2 * tmp_idx + 1
            else:
                tmp_idx = 2 * tmp_idx
        contribution_idx = self.n * tmp_idx + idx
        if normalized:
            return self.contributions[contribution_idx] / self.global_max_fitness
        else:
            return self.contributions[contribution_idx]

    def get_fitness_genome(self, genome: np.ndarray, normalized: bool = True) -> float:
        """
        get_fitness_genome(self, genome: np.ndarray, normalized: bool = True) -> float

        Returns the fitness of a genome.

        Parameters:
        genome (np.ndarray): The genome for which the fitness should be computed.
        normalized (bool, optional): Indicates whether the function returns the absolute or the normalized fitness of the genome. Defaults to True.

        Returns:
        float: The fitness of the genome.
        """
        self.check_genome(genome)
        perf: float = 0
        for i in range(self.n):
            perf += self.get_fitness_gene(i, genome, normalized=normalized)
        return perf / self.n

    def get_global_max(self):
        """
        get_global_max(self) -> Tuple[np.ndarray, float]

        Loops over all 2^N - 1 solutions genomes and computes the corresponding fitness. Returns the position of the global maximum in the landscape (as a bitstring) and the corresponding max performance.

        Parameters:
        None

        Returns:
        Tuple[np.ndarray, float]: A tuple containing the position of the global maximum in the landscape as a bitstring and the corresponding max performance.
        """
        tmp_pos = []
        tmp_fitness: float = 0.0
        for i in range((2**self.n) - 1):
            if (
                self.get_fitness_genome(
                    self.convert_number_to_bin_nparray(i), normalized=False
                )
                > tmp_fitness
            ):
                tmp_fitness = self.get_fitness_genome(
                    self.convert_number_to_bin_nparray(i), normalized=False
                )
                tmp_pos = self.convert_number_to_bin_nparray(i)
        return tmp_pos, tmp_fitness

    def recompute_contributions(self, correlation: float = 0):
        """
        recompute_contributions(self, correlation: float = 0) -> None

        Recomputes the contributions attribute of the object with either random uniform or correlated values.

        Parameters:
        correlation (float): The correlation coefficient between the new and old values of the contributions attribute. A value of 1 will result in no change to the attribute, -1 will result in the attribute being inverted, and 0 will result in the attribute being replaced with random uniform values. A value between -1 and 1 will result in the attribute being replaced with correlated values.

        Returns:
        None
        """
        # method to compute bivariate uniform data is taken from: https://www.tandfonline.com/doi/full/10.1080/03610926.2012.700373
        # see also: https://wernerantweiler.ca/blog.php?item=2020-07-05

        if correlation == 1:
            pass
        elif correlation == -1:
            for i in range(len(self.contributions)):
                self.contributions[i] = 1 - self.contributions[i]
        elif correlation == 0:
            for i in range(len(self.contributions)):
                self.contributions[i] = np.random.uniform(0, 1)
        else:
            # compute shape parameter for the Beta distribution
            a = 0.5 * (math.sqrt((49 + correlation) / (1 + correlation)) - 5)
            # update contributions with correlated values
            for i in range(len(self.contributions)):
                x = self.contributions[i]
                v = np.random.uniform(0, 1)
                w = np.random.beta(a, 1)
                if v < 0.5:
                    self.contributions[i] = abs(w - x)
                else:
                    self.contributions[i] = 1 - abs(1 - w - x)

        self.global_max_position, self.global_max_fitness = self.get_global_max()

    def get_contributions(self) -> np.ndarray:
        """
        get_contributions(self) -> np.ndarray

        Returns the contributions attribute of the object.

        Parameters:
        None

        Returns:
        np.ndarray: The contributions attribute of the object.
        """
        return self.contributions


class Agent(Person):
    def __init__(
        self,
        agent_idx: int,
        dependencymap: np.array,
        agents: int,
        decision_making_mode: str,
        information_exchange_mode: str,
        search_mode: str,
        min_genes: int,
        max_genes: int,
        error_mean: float = 0.0,
        error_std: float = 0.0,
        incentive_parameter: float = 0.5,
        number_of_proposals: int = 2,
    ):
        super().__init__(
            dependencymap,
            agents,
            decision_making_mode,
            information_exchange_mode,
            search_mode,
            error_mean,
            error_std,
            incentive_parameter,
            number_of_proposals,
        )
        self.responsibility = np.zeros(self.n)
        self.genes = set()
        self.information_state = dict()
        self.proposal_weight = self.set_proposal_weights()
        self.min_genes = min_genes
        self.max_genes = max_genes
        self.interdependence_beliefs = self.initialize_inderdependence_observations()
        self.learned_interdependencies = self.reset_learned_interdependencies()
        self.agent_idx = agent_idx

    def __str__(self):
        return f"Agent, operates on landscape with with N = {self.n} genes and K = {self.k} interdependencies, makes errors with mean {self.error_mean} and std {self.error_std}"

    @property
    def responsibility(self):
        return self._responsibility

    @responsibility.setter
    def responsibility(self, var: np.array):
        self._responsibility = var

    @property
    def agent_idx(self):
        return self._agent_idx

    @agent_idx.setter
    def agent_idx(self, var: int):
        self._agent_idx = var

    @property
    def genes(self):
        return self._genes

    @genes.setter
    def genes(self, var: set):
        self._genes = var

    @property
    def information_state(self):
        return self._information_state

    @information_state.setter
    def information_state(self, var: dict):
        self._information_state = var

    @property
    def proposal_weight(self):
        return self._proposal_weight

    @proposal_weight.setter
    def proposal_weight(self, var: dict):
        self._proposal_weight = var

    @property
    def min_genes(self):
        return self._min_genes

    @min_genes.setter
    def min_genes(self, var: int):
        self._min_genes = var

    @property
    def max_genes(self):
        return self._max_genes

    @max_genes.setter
    def max_genes(self, var: int):
        self._max_genes = var

    @property
    def interdependence_beliefs(self):
        return self._interdependence_beliefs

    @interdependence_beliefs.setter
    def interdependence_beliefs(self, var: dict):
        self._interdependence_beliefs = var

    @property
    def learned_interdependencies(self):
        return self._learned_interdependencies

    @learned_interdependencies.setter
    def learned_interdependencies(self, var: dict):
        self._learned_interdependencies = var

    @classmethod
    def get(
        cls,
        agent_idx: int,
        dependencymap: np.ndarray,
        agents: int,
        decision_making_mode: str,
        information_exchange_mode: str,
        search_mode: str,
        error_mean: float,
        error_std: float,
        incentive_parameter: float,
    ) -> object:
        return cls(
            agent_idx,
            dependencymap,
            agents,
            decision_making_mode,
            information_exchange_mode,
            search_mode,
            error_mean,
            error_std,
            incentive_parameter,
        )

    @staticmethod
    def flip_bit(var: int) -> int:
        """
        flip_bit(var: int) -> int

        Flips the value of a binary bit.

        Parameters:
        var (int): The binary bit to be flipped. Must be 0 or 1.

        Returns:
        int: The flipped binary bit. If var is 0, returns 1. If var is 1, returns 0.

        Raises:
        ValueError: If var is not 0 or 1.
        """
        if var == 0 or var == 1:
            if var == 0:
                return 1
            else:
                return 0
        else:
            raise ValueError("Bit must be one or zero")

    def get_utility(
        self, landscape: object, proposal: dict, estimate: bool = False
    ) -> float:
        """
        Calculates the utility of a proposal for a given landscape.
        If `estimate` is True, the utility is calculated by summing the normalized fitness contributions
        of the individual genes in the proposal, weighted by the proposal weight.
        If `estimate` is False, an error is raised.
        The utility is also adjusted by the incentive parameter and a random normal error.

        :param landscape: an object representing the fitness landscape
        :param proposal: a dictionary containing the proposed values for each gene in the genome
        :param estimate: a boolean indicating whether to estimate the utility or raise an error
        :param social: a boolean indicating whether the utility is to be computed for social search (True) or individual search (False)

        :return: the utility of the proposal
        """
        if estimate == True:

            tmp_utility_own: float = 0
            tmp_utility_residual: float = 0
            # print(f"proposal weights are {self.proposal_weight}")
            # print(f"responsibility is {self.genes}")
            # print(f"proposals are {proposal}")
            # print(f"information state is {self.information_state}")

            for i in range(self.number_of_proposals):
                # compute entire genome as an np.array
                tmp_genome = np.zeros(self.n)

                for j in range(self.n):
                    if j in self.genes:
                        tmp_genome[j] = proposal[j]
                    else:
                        tmp_genome[j] = self.information_state[i][j]

                # print(f"tmp genome for utility is {tmp_genome}")

                for j in range(self.n):
                    if j in self.genes:
                        tmp_utility_own += (
                            landscape.get_fitness_gene(j, tmp_genome, normalized=True)
                            * self.proposal_weight[i]
                        )
                    else:
                        tmp_utility_residual += (
                            landscape.get_fitness_gene(j, tmp_genome, normalized=True)
                            * self.proposal_weight[i]
                        )

        else:
            print("Specify way to compute utility from position")

        tmp_utility_own = tmp_utility_own / len(self.genes)
        tmp_utility_residual = tmp_utility_residual / (self.n - len(self.genes))

        utility = (
            self.incentive_parameter * tmp_utility_own
            + (1 - self.incentive_parameter) * tmp_utility_residual
            + np.random.normal(self.error_mean, self.error_std)
        )

        return utility

    def get_current_config(self) -> dict:
        """
        Returns a dictionary of the current configuration of the agent's genes

        :return: dictionary containing the current configuration of the agent's genes
        """

        current_config = dict()

        for gene in list(self.genes):
            current_config[gene] = self.position[gene]

        # print(f"current config is {current_config}")

        return current_config

    def get_alternative_positions(self, landscape: object, alternatives: int) -> dict:
        """
        Returns a dictionary of possible alternative configurations for the agent's area of responsibility, along with their estimated utility values.
        The dictionary includes the current configuration (status quo) as the last option.

        Parameters:
        landscape (object): The landscape object containing the utility function.
        alternatives (int): The number of alternative configurations to generate.

        Returns:
        dict: A dictionary where each key is an integer representing an alternative configuration, and each value is a dictionary containing the "proposal" (the alternative configuration) and "utility" (the estimated utility of the alternative configuration).
        """

        # create empty dict to store options for movements
        options = dict()
        # make copy of genes in the agents responsibility
        tmp_genes = copy.deepcopy(self.genes)

        # loop over number of alternatives
        for i in range(alternatives):
            # get the current configuation in the agent's area of responsibility
            proposal = self.get_current_config()
            # randomly select a bit to flip
            tmp_bit = rnd.choice(list(tmp_genes))

            # remove that bit from the pool of bits
            tmp_genes.remove(tmp_bit)

            # flip the bit
            if proposal[tmp_bit] == 0:
                proposal[tmp_bit] = 1
            else:
                proposal[tmp_bit] = 0

            options[i] = dict()
            options[i]["proposal"] = proposal
            options[i]["utility"] = self.get_utility(landscape, proposal, estimate=True)

        # last option is the status quo
        # store status-quo plus corresponding utiltiy as the last option
        options[alternatives] = dict()
        options[alternatives]["proposal"] = self.get_current_config()

        options[alternatives]["utility"] = self.get_utility(
            landscape, self.get_current_config(), estimate=True
        )

        # print(f"current configuration is {self.get_current_config()}")
        # print(f"options are {options}")

        # return list of options
        return options

    def hillclimbing(self, landscape: object, alternatives: int = 1) -> dict:
        """
        Returns a dictionary of proposed configurations using the hill climbing optimization method.
        The function generates alternative configurations using the `get_alternative_positions` method and selects the ones with the highest estimated utility. The number of proposals returned is determined by the `number_of_proposals` attribute of the object.

        Parameters:
        landscape (object): The landscape object containing the utility function.
        alternatives (int): The number of alternative configurations to generate. Default is 1.

        Returns:
        dict: A dictionary where each key is an integer representing a proposed configuration, and each value is the corresponding proposed configuration as a numpy array.
        """

        options = self.get_alternative_positions(landscape, alternatives)
        # print(f"options are {options}")

        # print(f"options are {options}")

        proposals = dict()
        keys = list(options.keys())
        # keys.remove("genes")
        # print(f"keys are {keys}")

        for i in range(self.number_of_proposals):
            proposal_key = ""
            proposal_utility = 0
            for key in keys:
                if options[key]["utility"] > proposal_utility:
                    proposal_key = key
                    proposal_utility = options[key]["utility"]
            proposals[i] = options[proposal_key]["proposal"]
            keys.remove(proposal_key)

        # print(f"proposals are {proposals}")

        return proposals

    def get_utility_social_search(
        self,
        landscape: object,
        own_options: dict,
        partner_options: dict,
        number_of_discoveries: int,
    ) -> dict:

        """Compute utilities of joint options.
        Parameters
        landscape : object
            Object representing the landscape on which to compute utilities.
        own_options : dict
            Dictionary of options for one individual.
        partner_options : dict
            Dictionary of options for the other individual.
        number_of_discoveries : int
            Number of discoveries made by agents to consider.

        Returns
        dict
            Dictionary of joint options, with keys 0 to `number_of_discoveries`+1 representing
            the different options and values being dictionaries containing the
            concatenated options and computed utilities for each individual.
        """

        joint_proposals = self.concatenate_discoveries_social_search(
            own_options, partner_options, number_of_discoveries
        )

        for k in range(number_of_discoveries + 1):

            tmp_utility_own: float = 0
            tmp_utility_residual: float = 0
            proposal = joint_proposals[k]["proposal"]
            tmp_genes = list(proposal.keys())

            for i in range(self.number_of_proposals):

                # compute entire genome as an np.array
                tmp_genome = np.zeros(self.n)

                for j in range(self.n):
                    if j in tmp_genes:
                        tmp_genome[j] = proposal[j]
                    else:
                        tmp_genome[j] = self.information_state[i][j]

                # print(f"tmp genome is {tmp_genome}")

                for j in range(self.n):
                    if j in self.genes:
                        tmp_utility_own += (
                            landscape.get_fitness_gene(j, tmp_genome, normalized=True)
                            * self.proposal_weight[i]
                        )
                    else:
                        tmp_utility_residual += (
                            landscape.get_fitness_gene(j, tmp_genome, normalized=True)
                            * self.proposal_weight[i]
                        )

            tmp_utility_own = tmp_utility_own / len(self.genes)
            tmp_utility_residual = tmp_utility_residual / (self.n - len(self.genes))

            utility = (
                self.incentive_parameter * tmp_utility_own
                + (1 - self.incentive_parameter) * tmp_utility_residual
                + np.random.normal(self.error_mean, self.error_std)
            )

            joint_proposals[k]["utility"] = utility

        # print(f"joint alternatives plus utility are {joint_proposals}")
        return joint_proposals

    def concatenate_discoveries_social_search(
        self, own_options: dict, partner_options: dict, alternatives: int
    ) -> dict:

        """
        Concatenate two dictionaries of options.

        Parameters
        own_options : dict
            Dictionary of options for one individual.
        partner_options : dict
            Dictionary of options for the other individual.
        alternatives : int
            Number of alternatives to consider.

        Returns
        dict
            Dictionary of joint options, with keys 0 to `alternatives` + 1 representing
            the different options and values being dictionaries containing the
            concatenated options for each individual.
        """

        joint_options = dict()
        # loop over alternatives plus 1 (for status quo)
        for i in range(alternatives + 1):
            joint_options[i] = dict()
            joint_options[i]["proposal"] = {
                **own_options[i]["proposal"],
                **partner_options[i]["proposal"],
            }
            joint_options[i]["proposal_own"] = own_options[i]["proposal"]
            joint_options[i]["proposal_partner"] = partner_options[i]["proposal"]

        # print("own options")
        # print(own_options)
        # print("partner options")
        # print(partner_options)
        # print("joint options")
        # print(f"joint alternatives are {joint_options}")

        return joint_options

    def social_hillclimbing(
        self, own_alternatives: dict, partner_alternatives: dict, alternatives: int
    ) -> dict:
        """
        Perform social hillclimbing to find the best proposals for both the self and partner agents.

        Parameters:
        - own_alternatives (dict): A dictionary of alternatives for the self agent, where each key is an alternative identifier, and the value is a dictionary containing the following keys:
            - "proposal_own" (int): The proposal made by the self agent for the given alternative.
            - "proposal_partner" (int): The proposal made by the partner agent for the given alternative.
            - "utility" (float): The utility of the given alternative for the self agent.
        - partner_alternatives (dict): A dictionary of alternatives for the partner agent, where each key is an alternative identifier, and the value is a dictionary containing the following keys:
            - "proposal_own" (int): The proposal made by the self agent for the given alternative.
            - "proposal_partner" (int): The proposal made by the partner agent for the given alternative.
            - "utility" (float): The utility of the given alternative for the self agent.
         - alternatives (int): The number of alternatives to consider.

        Returns:
        - A tuple containing two dictionaries, where the first dictionary is a mapping of indices to proposals made by the self agent, and the second dictionary is a mapping of indices to proposals made by the partner agent. Both dictionaries are sorted in decreasing order of utility.
        """

        for i in range(alternatives + 1):
            own_alternatives[i]["utility"] = (
                own_alternatives[i]["utility"] + partner_alternatives[i]["utility"]
            ) / 2

        # print(f"decision base is {own_alternatives}")

        proposals_own = dict()
        proposals_partner = dict()
        keys = list(own_alternatives.keys())

        for i in range(self.number_of_proposals):
            proposal_key = ""
            proposal_utility = 0
            for key in keys:
                if own_alternatives[key]["utility"] > proposal_utility:
                    proposal_key = key
                    proposal_utility = own_alternatives[key]["utility"]

            proposals_own[i] = own_alternatives[proposal_key]["proposal_own"]
            proposals_partner[i] = own_alternatives[proposal_key]["proposal_partner"]
            keys.remove(proposal_key)

        # print(f"final proposals: own -> {proposals_own} partner -> {proposals_partner}")

        return proposals_own, proposals_partner

    def simulated_annealing(
        self,
        landscape: object,
        initial_temperature: float,
        alpha: float,
        time: int,
        method: str = "exp_multiplicative",
    ) -> dict:
        """
        Returns a dictionary of proposed configurations using the simulated annealing optimization method.
        The function generates one alternative configuration using the `get_alternative_positions` method and selects it with a probability determined by the difference in estimated utility compared to the current configuration and the current temperature. The probability of selecting the alternative configuration decreases as the temperature decreases according to a cooling schedule specified by `method`. The number of proposals returned is always 2, with the first being the chosen proposal and the second being the current configuration.

        Parameters:
        landscape (object): The landscape object containing the utility function.
        initial_temperature (float): The initial temperature for the simulated annealing process.
        alpha (float): A hyperparameter for the cooling schedule.
        time (int): The current time step in the simulated annealing process.
        method (str): The cooling schedule to use. Can be "exp_multiplicative" for exponential multiplicative cooling or "log_multiplicative" for log multiplicative cooling. Default is "exp_multiplicative".

        Returns:
        dict: A dictionary where each key is an integer representing a proposed configuration, and each value is the corresponding proposed configuration as a numpy array.
        """

        options = self.get_alternative_positions(landscape, 1)
        proposal_key = ""
        proposals = dict()

        # difference = performance(current) - performance(next)
        difference = options[1]["utility"] - options[0]["utility"]

        # attention: this is a maximization problem, in the simulated annealing literature, minimization problems are addressed mainly
        # see https://link.springer.com/content/pdf/10.1023/B:COAP.0000044187.23143.bd.pdf?pdf=button%20sticky

        # if the new genome leads to higher utility than the current genome
        if difference <= 0:
            # proposal ranked first is the new genome
            proposal_key = 0
            prob = 0
        # otherwise
        else:
            if method == "exp_multiplicative":
                # exponential multiplicative cooling taken from Kirkpatrick, Gelatt and Vecchi (1983), see also http://what-when-how.com/artificial-intelligence/a-comparison-of-cooling-schedules-for-simulated-annealing-artificial-intelligence/
                # alpha usually between 0.8 and 0.9
                temperature = initial_temperature * alpha**time
            elif method == "log_multiplicative":
                # log multiplicative cooling taken from Aarts, E.H.L. & Korst, J., 1989, see also http://what-when-how.com/artificial-intelligence/a-comparison-of-cooling-schedules-for-simulated-annealing-artificial-intelligence/
                # alpha > 1
                temperature = initial_temperature / (1 + alpha * math.log(1 + time))

            prob = math.exp(-difference / temperature)
            # probability of picking the new option is prob, probatility of picking current option is 1-prob
            proposal_key = np.random.choice([0, 1], 1, p=[prob, 1 - prob])[0]

        # set up dict with ordered proposals
        proposals[0] = options[proposal_key]["proposal"]
        keys = list(options.keys())
        keys.remove(proposal_key)
        if len(keys) == 1:
            proposals[1] = options[keys[0]]["proposal"]
        else:
            raise KeyError("Wrong number of keys for simulated annealing")

        # print(f"annealing options are {options}")
        # print(f"difference is {difference}")
        # print(f"propbability of suboptimal move is {prob}")
        # print(f"proposal ranked first is {proposal_key}")
        # print(f"proposals are {proposals}")

        return proposals

    def social_simulated_annealing(
        self,
        own_alternatives: dict,
        partner_alternatives: dict,
        alternatives: int,
        initial_temperature: float,
        alpha: float,
        time: int,
        method: str = "exp_multiplicative",
    ) -> dict:

        """
        Run the social simulated annealing algorithm on the given alternatives.

        Parameters:
        own_alternatives (dict): A dictionary containing the alternatives available to the caller and their associated data.
        partner_alternatives (dict): A dictionary containing the alternatives available to the partner and their associated data.
        alternatives (int): The number of alternatives available.
        initial_temperature (float): The initial temperature for the simulated annealing algorithm.
        alpha (float): The cooling rate for the simulated annealing algorithm.
        time (int): The current time step.
        method (str, optional): The cooling schedule to use. Defaults to "exp_multiplicative".

        Returns:
        dict: A dictionary containing the proposals for both the caller and the partner, ranked in order of preference.
        """

        # compute average utility and store it in own_alternatives
        for i in range(alternatives + 1):
            own_alternatives[i]["utility"] = (
                own_alternatives[i]["utility"] + partner_alternatives[i]["utility"]
            ) / 2

        # alternatives and average utility are now stored in "own alternatives"

        proposals_own = dict()
        proposals_partner = dict()
        keys = list(own_alternatives.keys())

        # difference = performance(current) - performance(next)
        difference = own_alternatives[1]["utility"] - own_alternatives[0]["utility"]

        # attention: this is a maximization problem, in the simulated annealing literature, minimization problems are addressed mainly
        # see https://link.springer.com/content/pdf/10.1023/B:COAP.0000044187.23143.bd.pdf?pdf=button%20sticky

        # if the new genome leads to higher utility than the current genome
        if difference <= 0:
            # proposal ranked first is the new genome
            proposal_key = 0
            prob = 0
        # otherwise
        else:
            if method == "exp_multiplicative":
                # exponential multiplicative cooling taken from Kirkpatrick, Gelatt and Vecchi (1983), see also http://what-when-how.com/artificial-intelligence/a-comparison-of-cooling-schedules-for-simulated-annealing-artificial-intelligence/
                # alpha usually between 0.8 and 0.9
                temperature = initial_temperature * alpha**time
            elif method == "log_multiplicative":
                # log multiplicative cooling taken from Aarts, E.H.L. & Korst, J., 1989, see also http://what-when-how.com/artificial-intelligence/a-comparison-of-cooling-schedules-for-simulated-annealing-artificial-intelligence/
                # alpha > 1
                temperature = initial_temperature / (1 + alpha * math.log(1 + time))

            prob = math.exp(-difference / temperature)
            # probability of picking the new option is prob, probatility of picking current option is 1-prob
            proposal_key = np.random.choice([0, 1], 1, p=[prob, 1 - prob])[0]

        # set up dict with ordered proposals
        proposals_own[0] = own_alternatives[proposal_key]["proposal_own"]
        proposals_partner[0] = own_alternatives[proposal_key]["proposal_partner"]

        keys = list(own_alternatives.keys())
        keys.remove(proposal_key)
        if len(keys) == 1:
            proposals_own[1] = own_alternatives[keys[0]]["proposal_own"]
            proposals_partner[1] = own_alternatives[keys[0]]["proposal_partner"]
        else:
            raise KeyError("Wrong number of keys for simulated annealing")

        # print(f"annealing options are {own_alternatives}")
        # print(f"difference is {difference}")
        # print(f"propbability of suboptimal move is {prob}")
        # print(f"proposal ranked first is {proposal_key}")
        # print(f"own proposals are {proposals_own}")
        # print(f"partner proposals are {proposals_partner}")

        return proposals_own, proposals_partner

    def get_responsibility(self):
        """
        Returns the area of responsibility and genes of the object.

        Returns:
        tuple: A tuple where the first element is a list of the indices of the area of responsibility and the second element is a list of the indices of the genes in the area of responsibility.
        """
        return self.responsibility, self.genes

    def get_information_state(self):
        """
        Returns the information state of the object.
        The information state is a numpy array.

        Returns:
        np.array: The information state of the object.
        """
        return self.information_state

    def update_responsibility(self, var: np.array):
        """
        Updates the area of responsibility and genes of the object.

        Parameters:
        var (np.array): A binary numpy array where the indices with a value of 1 represent the new area of responsibility.

        Returns:
        None
        """
        self.responsibility = var
        tmp = set()
        for i in range(self.n):
            if var[i] == 1:
                tmp.add(i)
        self.genes = tmp

    def update_position(self, var: np.ndarray, initial: bool = False):
        """
        Updates the agent's information about the current position in the landscape
        :param var: numpy array containing a binary bitstring indicating the new position in the landscape
        :param initial: boolean variable, set to True when agents are initialized. If True, also the agent's information state (indicating what the agent's beliefs where the current position is) to the true position.
        """
        # update the position
        self.position = var

        # if initialization of agent objects, also initialize the agent's inforamtion state
        if initial == True:
            for i in range(self.number_of_proposals):
                self.information_state[i] = dict()
                for j in range(self.n):
                    self.information_state[i][j] = var[j]

    def update_information_state(self, var: dict):
        """
        Updates the information state of the object.
        The information state is a numpy array-

        Parameters:
        var (dict): A dictionary where each key is an integer representing a proposed configuration and each value is the corresponding proposed configuration as a numpy array. The dictionary should contain the same number of entries as the `number_of_proposals` attribute of the object.

        Returns:
        None
        """

        for i in range(self.number_of_proposals):
            for key in var[i].keys():
                self.information_state[i][key] = var[i][key]

    def set_proposal_weights(self):
        """
        Sets the weights for each proposal.
        The weights represent the relative importance of each proposal when making decisions. If the decision-making mode is decentralized, the weight for the first proposal is set to 1 and all other weights are set to 0. If the decision-making mode is centralized or hybrid, all weights are set to 1 divided by the number of proposals.

        Returns:
        dict: A dictionary where each key is an integer representing a proposal and each value is the weight for that proposal.
        """

        # initialize dict object
        weights = dict()

        # loop over proposals an agent can make
        for i in range(self.number_of_proposals):
            # if decisions are made decentrally, set weight for first proposal to 1, all other weights to 0
            if self.decision_making_mode == "decentralized":
                if i == 0:
                    weights[i] = 1
                else:
                    weights[i] = 0
            # otherwise set all weights to 1 / number of proposals
            else:
                weights[i] = 1 / self.number_of_proposals

        return weights

    def vote_on_bitstrings(self, landscape: object, bitstrings: dict) -> dict:
        """
        Votes on a set of proposals provided by the coordinator.
        For each proposal, the agent compares the utility of the proposal with the current utility of its own configuration. If the proposal has higher utility, the agent votes in favor (True), otherwise it votes against (False).

        Parameters:
        landscape (object): The landscape object containing the performance landscape
        bitstrings (dict): A dictionary where each key is an integer representing a proposal and each value is the corresponding proposal as a numpy array.

        Returns:
        dict: A dictionary where each key is an integer representing a proposal and each value is the vote for that proposal (True or False).

        """
        # initialize dict to store the agent's votes
        voting = dict()

        # loop over all proposals
        for i in range(self.number_of_proposals):
            if self.get_utility_full_bitstring(
                landscape, bitstrings[i]
            ) > self.get_utility_full_bitstring(landscape, self.get_position()):
                voting[i] = True
            else:
                voting[i] = False

        return voting

    def get_utility_full_bitstring(
        self, landscape: object, bitstring: np.array, estimate: bool = True
    ) -> float:
        """
        Returns the utility of a full bitstring.
        The utility is computed as the average normalized fitness of the genes in the object's area of responsibility, weighted by the incentive parameter, plus the average normalized fitness of the genes outside the object's area of responsibility, weighted by 1 minus the incentive parameter, plus a normally distributed error term.

        Parameters:
        landscape (object): The landscape object containing the performance landscape.
        bitstring (np.array): An n-dimensional binary bitstring stored in a numpy array.

        Returns:
        float: The utility of the bitstring.
        """
        tmp_utility_own: float = 0.0
        tmp_utility_residual: float = 0.0
        utility: float = 0.0

        for i in range(self.n):
            if i in self.genes:
                tmp_utility_own += landscape.get_fitness_gene(
                    i, bitstring, normalized=True
                )
            else:
                tmp_utility_residual += landscape.get_fitness_gene(
                    i, bitstring, normalized=True
                )

        # take average fitness
        tmp_utility_own = tmp_utility_own / len(self.genes)
        tmp_utility_residual = tmp_utility_residual / (self.n - len(self.genes))

        if estimate == True:
            utility = (
                self.incentive_parameter * tmp_utility_own
                + (1 - self.incentive_parameter) * tmp_utility_residual
                + np.random.normal(self.error_mean, self.error_std)
            )
        else:
            utility = (
                self.incentive_parameter * tmp_utility_own
                + (1 - self.incentive_parameter) * tmp_utility_residual
            )

        # print(utility)
        return utility

    def initialize_inderdependence_observations(self):
        """
        Initializes the observations of interdependencies between genes.
        For each pair of genes (i, j), this function initializes the number of positive observations (alpha) and negative observations (beta). If i and j are the same, alpha is set to 1 and beta is set to 0, resulting in a belief about the interdependence of 1 (100%). If i and j are different, both alpha and beta are set to 1, resulting in a belief about the interdependence of 0.5 (50%).

        Returns:
        dict: A dictionary where each key is a tuple (i, j) representing a pair of genes and each value is a dictionary with keys "alpha" and "beta" representing the number of positive and negative observations, respectively.
        """

        beliefs = dict()

        for i in range(self.n):
            for j in range(self.n):
                beliefs[(i, j)] = dict()
                beliefs[(i, j)]["alpha"] = 1
                if i == j:
                    beliefs[(i, j)]["beta"] = 0
                else:
                    beliefs[(i, j)]["beta"] = 1

        return beliefs

    def reset_learned_interdependencies(self):
        """
        Resets the dictionary storing the learned interdependencies.
        This function resets or initializes a dictionary where each key is a tuple (i, j) representing a pair of genes and each value is a dictionary with keys "alpha" and "beta" representing the number of positive and negative observations, respectively. All values are set to zero.

        Returns:
        dict: The reset or initialized dictionary.
        """

        learnings = dict()

        for i in range(self.n):
            for j in range(self.n):
                learnings[(i, j)] = dict()
                learnings[(i, j)]["alpha"] = 0
                learnings[(i, j)]["beta"] = 0

        return learnings

    def update_interdependence_observations(
        self, landscape: object, new_position: np.array, mode: str
    ):
        """
        Updates observations about interdependencies between genes.

        :param landscape: landscape object that contains the NK performance landscape used to compute the fitness of genes
        :param new_position: numpy array representing the new position of the agent
        :param mode: string indicating the mode of updating interdependence observations. Can be "own" or "full".
            "own": agent uses genes in own area of responsibility to perform the update, learned information is not shared with other agents
            "full": agent uses the entire bitsting to perform an update, learned information is not shared with other agents

        :return: None
        """

        # reset the learnings to values of all zero
        self.learned_interdependencies = self.reset_learned_interdependencies()

        # print("Agent learns")

        if mode == "own":
            # loop over all genes in the own area of responsibility
            for flipped_gene in self.genes:
                # if the genes was flipped when the position was changed from the old to the new position
                if self.position[flipped_gene] != new_position[flipped_gene]:
                    # loop over all genes in the own area of responsibility
                    for affected_gene in self.genes:
                        # check if flipping flipped_gene affects the performance contributions of other genes as well
                        if flipped_gene != affected_gene:
                            # if so, increase alpha by 1
                            if landscape.get_fitness_gene(
                                affected_gene, self.position, normalized=True
                            ) != landscape.get_fitness_gene(
                                affected_gene, new_position, normalized=True
                            ):
                                self.interdependence_beliefs[
                                    (flipped_gene, affected_gene)
                                ]["alpha"] += 1
                                self.learned_interdependencies[
                                    (flipped_gene, affected_gene)
                                ]["alpha"] += 1
                                # print("alpha increased by 1")
                            else:
                                self.interdependence_beliefs[
                                    (flipped_gene, affected_gene)
                                ]["beta"] += 1
                                self.learned_interdependencies[
                                    (flipped_gene, affected_gene)
                                ]["beta"] += 1
            # print(f"in mode {mode}, agent learned interdependencies of {self.learned_interdependencies}")

        elif mode == "full":
            # loop over all genes
            for flipped_gene in range(self.n):
                # if the genes was flipped when the position was changed from the old to the new position
                if self.position[flipped_gene] != new_position[flipped_gene]:
                    # loop over all genes in the own area of responsibility
                    # print(f"bit {flipped_gene} has changed")
                    for affected_gene in self.genes:
                        # check if flipping flipped_gene affects the performance contributions of other genes in the own area of responsibility
                        if flipped_gene != affected_gene:
                            # if so, increase alpha by 1
                            if landscape.get_fitness_gene(
                                affected_gene, self.position, normalized=True
                            ) != landscape.get_fitness_gene(
                                affected_gene, new_position, normalized=True
                            ):
                                self.interdependence_beliefs[
                                    (flipped_gene, affected_gene)
                                ]["alpha"] += 1
                                self.learned_interdependencies[
                                    (flipped_gene, affected_gene)
                                ]["alpha"] += 1
                                # print(f"alpha increased by one ({flipped_gene} / {affected_gene})")
                            # otherwise increase beta by 1
                            else:
                                self.interdependence_beliefs[
                                    (flipped_gene, affected_gene)
                                ]["beta"] += 1
                                self.learned_interdependencies[
                                    (flipped_gene, affected_gene)
                                ]["beta"] += 1
                                # print(f"beta increased by one ({flipped_gene} / {affected_gene})")
            # print(f"in mode {mode}, agent learned interdependencies of {self.learned_interdependencies}")

        else:
            print("Mode to update observations about interdependencies not specified")

    def get_learned_interdependencies(self) -> dict:
        """
        Returns the observations about interdependencies that were made in the current period.
        :return: dict object containing observations about interdependencies made in the current period.
        """
        return self.learned_interdependencies

    def social_learning_interdependencies(self, learnings: dict):
        """
        Updates the observations about interdependencies for this agent based on the learnings of another agent.

        Parameters:
        learnings: A dictionary containing the observations about interdependencies learned by another agent.

        Returns:
            None
        """

        for i in range(self.n):
            for j in range(self.n):
                self.interdependence_beliefs[(i, j)]["alpha"] += learnings[(i, j)][
                    "alpha"
                ]
                self.interdependence_beliefs[(i, j)]["beta"] += learnings[(i, j)][
                    "beta"
                ]

    def get_offer_allocation(
        self, landscape: object, mode: str, weight: float = 0.5
    ) -> dict:
        """
        Computes the genes an agents wants to offer to other agents in bottom-up task allocatioin
        :param landscape: landscape object that contains the NK performance landscape used to compute utility
        :param mode: str, specifies how the agent computes its offer
            performance: agent computes the offer based on the performance of the genes in its area of responsibility
            interdependencies: agent computes the offer based on the average internal belief on interdependencies of the genes in its area of responsibility
            utility: agent computes the offer based on the increase in utility that results from transferring the gene to another agent
            weighted_performance_interdependence: agent computes the offer based on a weighted sum of performance and interdependence
        :param weight: float, weight applied to the average internal belief on interdependencies
        :return: dict with two entries
            gene: gene that the agent wants to offer
            threshhold: minimum bid that other agents have to make to get the gene
        """

        offer = dict()
        offer["gene"]: int = False
        offer["threshold"]: float = 1

        # check if agents have enough genes so that they can offer
        if len(self.genes) > self.min_genes:
            # if bottom-up task allocation is performance-based
            if mode == "performance":

                # loop over all decisions in the agent's area of reponsibility
                for gene in self.genes:
                    # find the gene in the own area of responsibility that is associated with the minimum performance
                    estimate_performance = landscape.get_fitness_gene(
                        gene, self.position, normalized=True
                    )
                    if estimate_performance < offer["threshold"]:
                        offer["threshold"] = estimate_performance
                        offer["gene"] = gene

            elif mode == "interdependence":

                # loop over genes
                for gene in self.genes:
                    # find the gene with the minimum belief on average internal interdependence and store it in the offer dict
                    estimate_belief = self.get_mean_internal_belief(gene)
                    # print(f"current gene is {offer['gene']} with threshold {offer['threshold']}. Evaluated gene is {gene} with belief on interdependencies {estimate_belief}")
                    if estimate_belief < offer["threshold"]:
                        offer["threshold"] = estimate_belief
                        offer["gene"] = gene
                        # print(f"new gene is {gene}")

            elif mode == "utility":
                #
                # here, the threshold is zero because agents are offering the gene which results in the highest increase in their utility
                # even if they do not get a compensation payment for transferring a gene to another agents
                # they experience an increase in utility
                #
                offer["threshold"]: float = 0
                tmp_threshold: float = 1

                # loop over genes
                for gene in self.genes:
                    # find the gene with the minimum utility and store it in the offer dict
                    estimate_utility = 1 - self.estimate_utility_of_swapping(
                        landscape, gene, remove_gene=True
                    )
                    # if estimate_utility > tmp_threshold:
                    #     tmp_threshold= estimate_utility
                    #    offer["gene"] = gene
                    # print(f"current gene is {offer['gene']} with threshold {tmp_threshold}. Evaluated gene is {gene} with utility {estimate_utility}")
                    if estimate_utility < tmp_threshold:
                        tmp_threshold = estimate_utility
                        offer["gene"] = gene
                        # print(f"new gene is {gene}")

            elif mode == "weighted_performance_interdependence":

                # loop over all decisions in the agent's area of reponsibility
                for gene in self.genes:
                    # find the gene in the own area of responsibility that is associated with the minimum weighted sum of performance and interdependence
                    estimate_performance = landscape.get_fitness_gene(
                        gene, self.position, normalized=True
                    )
                    estimate_belief = self.get_mean_internal_belief(gene)

                    # compute weighted sum
                    weighted_sum = (
                        weight * estimate_belief + (1 - weight) * estimate_performance
                    )

                    # print(f"current gene: {offer['gene']}; threshold {offer['threshold']}. Evaluated gene is {gene}; belief on interdependencies {estimate_belief}; estimated performance {estimate_performance}. Weighted sum is {weighted_sum}")
                    # update gene that is offered accordingly so that the minimum of the weighted sum is offered
                    if weighted_sum < offer["threshold"]:
                        offer["threshold"] = weighted_sum
                        offer["gene"] = gene
                        # print(f"new gene is {gene}")

            elif mode == "weighted_utility_interdependence":

                offer["threshold"]: float = 1
                tmp_threshold: float = 1

                # loop over genes
                for gene in self.genes:
                    # find the gene in the own area of responsibility that is associated with the minimum weighted sum of utility and interdependence
                    estimate_utility = 1 - self.estimate_utility_of_swapping(
                        landscape, gene, remove_gene=True
                    )
                    estimate_belief = self.get_mean_internal_belief(gene)

                    # compute weighted sum
                    weighted_sum = (
                        weight * estimate_belief + (1 - weight) * estimate_utility
                    )

                    # print(f"current gene: {offer['gene']}; threshold {tmp_threshold}. Evaluated gene is {gene}; belief on interdependencies {estimate_belief}; estimated utility {estimate_utility}. Weighted sum is {weighted_sum}")

                    # update as follows
                    # if weighted sum is below the threshold
                    if weighted_sum < tmp_threshold:
                        # set weighted sum as new threshold
                        tmp_threshold = weighted_sum
                        # set the threshold to swap the task with another agent to weight * estimate belief
                        # special case for utility-based approaches here .. every utlity > 0 is acceptable, therefore, the threshold for utility would be zero
                        # (1-weight) * 0 is ommited here
                        offer["threshold"] = weight * estimate_belief
                        offer["gene"] = gene
                        # print(f"new gene is {gene}")

            else:
                sys.exit(
                    "Mode to compute offers in bottom-up task allocation not specified"
                )
        else:
            offer["gene"] = False

        # print(f"offer is {offer['gene']} at threshold {offer['threshold']}")
        return offer

    def get_bids_allocation(
        self,
        offers: dict,
        landscape: object,
        mode: str,
        error_mean: float = 0,
        error_std: float = 0,
        weight: float = 0.5,
    ) -> dict:
        """
        Computes bids for the offered genes from other agents.

        :param offers: dict of offers from other agents, in the form {<agent index>: {'gene': <int/False>, 'threshold': <float/False>}}
        :param landscape: landscape object that contains the NK performance landscape used to compute the bids
        :param mode: str, mode for computing bids. Can be 'performance', 'interdependencies', 'utility', or 'weighted_performance_interdependence'
        :param error_mean: float, mean of the normal distribution used to add error to the bids
        :param error_std: float, standard deviation of the normal distribution used to add error to the bids
        :param weight: float, weight used to compute the weighted sum of performance and interdependence when mode is 'weighted_performance_interdependence'

        :return: dict, with entries 'bid' for each agent index
        """

        bids = dict()
        bids_test = dict()

        # check if agent still have free capacities to get an additional gene
        if len(self.genes) < self.max_genes:
            # if bids are computed on the basis of performances
            if mode == "performance":
                # loop over agents
                for i in range(self.agents):
                    # initialize datastructure
                    bids[i] = dict()
                    # if the offer was not created by the same agent
                    if i != self.agent_idx and offers[i]["gene"] is not False:
                        # compute the bid and store it
                        estimate_performance = landscape.get_fitness_gene(
                            offers[i]["gene"], self.position, normalized=True
                        ) + np.random.normal(error_mean, error_std)
                        bids[i]["bid"] = estimate_performance
                    # otherwise submit false
                    else:
                        bids[i]["bid"] = False

            elif mode == "interdependence":

                # loop over agents
                for i in range(self.agents):
                    # initialize datastructure
                    bids[i] = dict()
                    # if the offer was not created by the same agent
                    if i != self.agent_idx and offers[i]["gene"] is not False:
                        # compute the bid and store it
                        estimate_belief = self.get_mean_internal_belief(
                            offers[i]["gene"]
                        )
                        bids[i]["bid"] = estimate_belief
                    # otherwise submit false
                    else:
                        bids[i]["bid"] = False

            elif mode == "utility":

                for i in range(self.agents):
                    # initialize datastructure
                    bids[i] = dict()
                    # if the offer was not created by the same agent
                    if i != self.agent_idx and offers[i]["gene"] is not False:
                        # compute the bid and store it
                        estimate_utility = self.estimate_utility_of_swapping(
                            landscape, offers[i]["gene"], add_gene=True
                        ) + np.random.normal(error_mean, error_std)
                        bids[i]["bid"] = estimate_utility
                    # otherwise submit false
                    else:
                        bids[i]["bid"] = False

            elif mode == "weighted_performance_interdependence":

                for i in range(self.agents):
                    # initialize datastructure
                    bids[i] = dict()
                    # if the offer was not created by the same agent
                    if i != self.agent_idx and offers[i]["gene"] is not False:
                        # compute the bid and store it
                        estimate_performance = landscape.get_fitness_gene(
                            offers[i]["gene"], self.position, normalized=True
                        ) + np.random.normal(error_mean, error_std)
                        estimate_belief = self.get_mean_internal_belief(
                            offers[i]["gene"]
                        )
                        bids[i]["bid"] = (
                            weight * estimate_belief
                            + (1 - weight) * estimate_performance
                        )
                    # otherwise submit false
                    else:
                        bids[i]["bid"] = False

            elif mode == "weighted_utility_interdependence":

                for i in range(self.agents):
                    # initialize datastructure
                    bids[i] = dict()
                    # if the offer was not created by the same agent
                    if i != self.agent_idx and offers[i]["gene"] is not False:
                        # compute the bid and store it
                        estimate_utility = self.estimate_utility_of_swapping(
                            landscape, offers[i]["gene"], add_gene=True
                        ) + np.random.normal(error_mean, error_std)
                        estimate_belief = self.get_mean_internal_belief(
                            offers[i]["gene"]
                        )
                        bids[i]["bid"] = (
                            weight * estimate_belief + (1 - weight) * estimate_utility
                        )
                    # otherwise submit false
                    else:
                        bids[i]["bid"] = False

            else:
                sys.exit(
                    "Mode to compute offers in bottom-up task allocation not specified"
                )

        # if there is no free capacity
        # submit no bid but False to inidcate that the agent does not participate
        else:
            # print("no free capacity")
            for i in range(self.agents):
                bids[i] = dict()
                bids[i]["bid"] = False

        return bids

    def get_interdependence_beliefs(self) -> dict:
        """
        Returns the current interdependence beliefs of the agent.
        :return: dict object
        """
        return self.interdependence_beliefs

    def get_mean_internal_belief(self, gene_of_interest: int) -> float:
        """
        Returns the mean belief of an agent on the interdependence of a gene with all other genes in the agent's area of responsibility
        :param gene_of_interest: index of the gene for which the mean belief on interdependence should be computed
        :return: float value between 0 and 1
        """

        tmp_belief: float = 0
        count: int = 0

        for gene in self.genes:
            if gene != gene_of_interest:
                a = self.interdependence_beliefs[(gene_of_interest, gene)]["alpha"]
                b = self.interdependence_beliefs[(gene_of_interest, gene)]["beta"]
                tmp_belief += beta.stats(a, b, moments="m")
                a = self.interdependence_beliefs[(gene, gene_of_interest)]["alpha"]
                b = self.interdependence_beliefs[(gene, gene_of_interest)]["beta"]
                tmp_belief += beta.stats(a, b, moments="m")
                count += 2

        return tmp_belief / count

    def estimate_utility_of_swapping(
        self,
        landscape: object,
        gene: int,
        remove_gene: bool = False,
        add_gene: bool = False,
    ) -> float:
        """
        Estimates the utility of a gene swap operation.

        :param landscape: Landscape object representing the problem domain.
        :param gene: Gene to be added or removed.
        :param remove_gene: Boolean indicating whether the gene should be removed.
        :param add_gene: Boolean indicating whether the gene should be added.
        :return: Estimate of the utility of the gene swap.
        """

        tmp_utility_own: float = 0.0
        tmp_utility_residual: float = 0.0
        est_utility: float = 0.0
        tmp_genes = copy.deepcopy(self.genes)

        if remove_gene:
            tmp_genes.remove(gene)
        elif add_gene:
            tmp_genes.add(gene)
        else:
            sys.exit("Specify whether gene should be removed or added!")

        for i in range(self.n):
            if i in tmp_genes:
                tmp_utility_own += landscape.get_fitness_gene(
                    i, self.position, normalized=True
                )
            else:
                tmp_utility_residual += landscape.get_fitness_gene(
                    i, self.position, normalized=True
                )

        tmp_utility_own = tmp_utility_own / len(tmp_genes)
        tmp_utility_residual = tmp_utility_residual / (self.n - len(tmp_genes))

        est_utility = (
            self.incentive_parameter * tmp_utility_own
            + (1 - self.incentive_parameter) * tmp_utility_residual
            + np.random.normal(self.error_mean, self.error_std)
        )

        actual_utility = self.get_utility_full_bitstring(landscape, self.position)
        # if remove_gene:
        # print(f"acutal utility of bitstring is {actual_utility}")
        # print(f"REMOVE: estimated utility after swapping is {est_utility}")
        # print(f"{gene}: returned: {est_utility - actual_utility}")
        # elif add_gene:
        # print(f"acutal utility of bitstring is {actual_utility}")
        # print(f"ADD: estimated utility after swapping is {est_utility}")
        # print(f"{gene}: returned: {est_utility - actual_utility}")

        return est_utility - actual_utility


class Coordinator(Person):
    def __init__(
        self,
        dependencymap: np.ndarray,
        agents: int,
        decision_making_mode: str,
        information_exchange_mode: str,
        search_mode: str,
        sequential_allocation: bool,
        error_mean: float = 0.0,
        error_std: float = 0.0,
        incentive_parameter: float = 0.5,
        number_of_proposals: int = 2,
    ):
        super().__init__(
            dependencymap,
            agents,
            decision_making_mode,
            information_exchange_mode,
            search_mode,
            error_mean,
            error_std,
            incentive_parameter,
            number_of_proposals,
        )
        self.sequential_allocation = sequential_allocation
        self.allocation = self.allocate_genes()
        self.agents_proposals = dict()
        self.position = self.compute_initial_position()
        self.reallocation = dict()

    @property
    def sequential_allocation(self):
        return self._sequential_allocation

    @sequential_allocation.setter
    def sequential_allocation(self, var: bool):
        self._sequential_allocation = var

    @property
    def allocation(self):
        return self._allocation

    @allocation.setter
    def allocation(self, var: np.array):
        self._allocation = var

    @property
    def agents_proposals(self):
        return self._agents_proposals

    @agents_proposals.setter
    def agents_proposals(self, var: np.array):
        self._agents_proposals = var

    @property
    def reallocation(self):
        return self._reallocation

    @reallocation.setter
    def reallocation(self, var: dict):
        self._reallocation = var

    @classmethod
    def get(
        cls,
        dependencymap: np.ndarray,
        agents: int,
        sequential_allocation: bool,
        decision_making_mode: str,
        information_exchange_mode: str,
        search_mode: str,
        error_mean: float,
        error_std: float,
        incentive_parameter: float,
    ) -> object:
        return cls(
            dependencymap,
            agents,
            sequential_allocation,
            decision_making_mode,
            information_exchange_mode,
            search_mode,
            error_mean,
            error_std,
            incentive_parameter,
        )

    def allocate_genes(self) -> np.array:
        """
        Initializes the task allocation for agents by allocating genes/tasks to agents.

        Returns:
        A two-dimensional numpy array where the first dimension is the number of agents, and the second dimension is the number of genes in genome.
        A 1 indicates that the gene is allocated to the corresponding agent, while a 0 indicates that it is not.

        Raises:
        ValueError: If the number of agents is greater than the number of genes.
        """

        if self.agents > self.n:
            raise ValueError(
                "Number of agents cannot be greater than the number of genes"
            )
        else:
            # initialize empty numpy array with 2 dimensions
            # first dimension: number of agents
            # second dimension: number of genes in genome
            task_allocation = np.zeros((self.agents, self.n))

            # if genes can be allocated symmetrically (i.e., each agent is responsible for the same number of tasks)
            if self.n % self.agents == 0:

                # compute genes per agent (except for the last agent, who will be responsible for the remaining tasks)
                genes_per_agent = math.floor(self.n / self.agents)

                # if sequential allocation
                if self.sequential_allocation == True:

                    # loop over all agents and allocate genes sequentially
                    for i in range(self.agents):
                        for j in range(genes_per_agent):
                            task_allocation[i, i * genes_per_agent + j] = 1

                # if random allocation
                else:
                    # initialize list of genes
                    genes = list(range(self.n))

                    # loop over all agents and allocate genes randomly
                    for i in range(self.agents):
                        for j in range(genes_per_agent):
                            tmp = rnd.choice(genes)
                            task_allocation[i, tmp] = 1
                            genes.remove(tmp)

            # if genes cannot be allocated symmetrically (i.e., the last agent will be responsible for the remaining tasks)
            else:

                # compute genes per agent (except for the last agent, who will be responsible for the remaining tasks)
                genes_per_agent = math.floor((self.n - 1) / (self.agents - 1))

                # if sequential allocation
                if self.sequential_allocation == True:
                    # loop over all except for the last agent and allocate tasks sequentially
                    for i in range(self.agents - 1):
                        for j in range(genes_per_agent):
                            task_allocation[i, i * genes_per_agent + j] = 1

                    # perform task allocation for the last agents (assign remaining genes)
                    for j in range(self.n - genes_per_agent * (self.agents - 1)):
                        task_allocation[self.agents - 1, -(j + 1)] = 1

                # if random allocaiton
                else:
                    # initialize list of genes
                    genes = list(range(self.n))

                    for i in range(self.agents - 1):
                        for j in range(genes_per_agent):
                            tmp = rnd.choice(genes)
                            task_allocation[i, tmp] = 1
                            genes.remove(tmp)

                    for item in genes:
                        task_allocation[self.agents - 1, item] = 1

            return task_allocation

    def get_allocation(self) -> np.array:
        """
        Returns the current gene allocation as a two-dimensional numpy array.
        The first dimension represents the agents, and the second dimension represents the genes.
        A value of 1 in the array indicates that the gene is allocated to the corresponding agent,
        while a value of 0 indicates that it is not.

        :return: two-dimensional numpy array with gene allocation
        """
        return self.allocation

    def compute_initial_position(self) -> np.array:
        """
        Generate a random binary position of length `n` for the agent to start at in the landscape
        :return: a numpy array representing the initial position of the agent in the landscape
        """
        # get random number
        rd = np.random.randint(0, 2**self.n - 1)

        # print(f"coordinator initial position is {self.convert_number_to_bin_nparray(rd)}")

        return self.convert_number_to_bin_nparray(rd)

    def update_proposals(self, agent: int, proposals: dict):

        """
        Update the proposals made by agents in the current round.
        If it is a new round, reset the proposals.
        :param agent: index of the agent that is submitting proposals
        :param proposals: proposals made by the agent
        """
        # if it is a new round of submitting proposals (if the agent is the first one, i.e., zero)
        if agent == 0:
            self.reset_proposals()

        self.agents_proposals[agent] = proposals
        # print(f"coordinator: Proposals are {self.agents_proposals}")

    def get_proposals(self) -> dict:
        """
        Returns a dictionary containing the proposals made by all agents in the current round of proposals.

        :return: dictionary with agent id as key and a dictionary of proposals as value
        """
        return self.agents_proposals

    def get_proposals_agent(self, agent: int) -> dict:
        """
        Returns the proposals made by a specific agent

        :param agent: index of the agent whose proposals should be returned
        :type agent: int
        :return: dictionary containing the proposals made by the agent
        :rtype: dict
        """

        return self.agents_proposals[agent]

    def reset_proposals(self):
        """
        Resets the proposals submitted by agents.
        """
        self.agents_proposals = dict()

    def make_decision(
        self,
        landscape: object,
        agents: object,
        mode: str,
        initial_temperature: float = 0,
        alpha: float = 0,
        time: int = 0,
        method: str = "exp_multiplicative",
    ):
        """
        Makes a decision about which alternative to choose based on the given mode.

        Parameters:
        landscape (object): The landscape object.
        agents (object): A list of agents.
        mode (str): The mode to use for decision making; one of 'decentralized', 'centralized_hillclimbing', 'centralized_simulated_annealing', or 'hybrid'.
        initial_temperature (float, optional): The initial temperature for the simulated annealing algorithm. Defaults to 0.
        alpha (float, optional): The cooling rate for the simulated annealing algorithm. Defaults to 0.
        time (int, optional): The current time step. Defaults to 0.
        method (str, optional): The cooling schedule to use. Defaults to "exp_multiplicative".

        Returns:
        None
        """

        # print(f"decision making mode is {self.decision_making_mode}")

        if mode == "decentralized":
            self.position = self.concatenate_proposals(0)
            # print(f"coordinator decentralized: position is {self.position}")

        elif mode == "centralized_hillclimbing":

            selected_alternative = 0
            tmp_utility = 0

            for i in range(self.number_of_proposals):
                if (
                    self.compute_utility(landscape, self.concatenate_proposals(i))
                    >= tmp_utility
                ):
                    tmp_utility = self.compute_utility(
                        landscape, self.concatenate_proposals(i)
                    )
                    selected_alternative = i

            self.position = self.concatenate_proposals(selected_alternative)
            # print(f"coordinator centralized: position is {self.position}")

        elif mode == "centralized_simulated_annealing":

            if initial_temperature == 0 or alpha == 0 or time == 0:
                sys.exit("Specify paramters for simualted annealing (centralized)")

            selected_alternative = 0
            utiltiy_discovery = self.compute_utility(
                landscape, self.concatenate_proposals(0)
            )
            utility_status_quo = self.compute_utility(
                landscape, self.concatenate_proposals(1)
            )

            difference = utility_status_quo - utiltiy_discovery

            # if the new genome leads to higher utility than the current genome
            if difference <= 0:
                # proposal ranked first is the new genome
                proposal_key = 0
                prob = 0
            # otherwise
            else:
                if method == "exp_multiplicative":
                    # exponential multiplicative cooling taken from Kirkpatrick, Gelatt and Vecchi (1983), see also http://what-when-how.com/artificial-intelligence/a-comparison-of-cooling-schedules-for-simulated-annealing-artificial-intelligence/
                    # alpha usually between 0.8 and 0.9
                    temperature = initial_temperature * alpha**time
                elif method == "log_multiplicative":
                    # log multiplicative cooling taken from Aarts, E.H.L. & Korst, J., 1989, see also http://what-when-how.com/artificial-intelligence/a-comparison-of-cooling-schedules-for-simulated-annealing-artificial-intelligence/
                    # alpha > 1
                    temperature = initial_temperature / (1 + alpha * math.log(1 + time))

                prob = math.exp(-difference / temperature)
                # probability of picking the new option is prob, probatility of picking current option is 1-prob
                proposal_key = np.random.choice([0, 1], 1, p=[prob, 1 - prob])[0]

            # print(f"difference is {difference}")
            # print(f"utility 0 is {utiltiy_discovery}")
            # print(f"utility 1 is {utility_status_quo}")
            # print(f"prob of suboptimal move is {prob}")
            # print(f"options are {self.agents_proposals}")
            # print(f"selected alternative is {proposal_key}")

            self.position = self.concatenate_proposals(proposal_key)

        elif mode == "hybrid":
            proposals_for_voting = dict()
            voting = dict()

            # create dict with all concatenated proposals, concatentation happens by rank
            for i in range(self.number_of_proposals):
                proposals_for_voting[i] = self.concatenate_proposals(i)

            # print(proposals_for_voting)

            # print(proposals_for_voting)

            # get the agents' feedback on the proposals (get their votes)
            for i in range(self.agents):
                voting[i] = agents[i].vote_on_bitstrings(
                    landscape, proposals_for_voting
                )

            # print(voting)
            # print(f"current position is {self.get_position()}")

            for i in range(self.number_of_proposals):
                selected = True
                for j in range(self.agents):
                    if voting[j][i] == False:
                        selected = False
                        break
                if selected == True:
                    self.position = proposals_for_voting[i]
                    # print(f"position updated with proposal {i}")
                    break

        else:
            sys.exit("Decision making mode not specified")

    def compute_utility(self, landscape: object, genome: np.array) -> float:
        """
        Compute the utility of a given genome on a given landscape

        :param landscape: object representing the landscape
        :param genome: numpy array representing the genome
        :return: float representing the utility of the genome on the landscape
        """
        # print(f"coordinator's utility is {landscape.get_fitness_genome(genome)}")

        return landscape.get_fitness_genome(genome)

    def concatenate_proposals(self, rank: int) -> np.array:
        """
        Returns the concatenation of proposals of all agents, based on the rank of the proposal.

        :param rank: rank of the proposal to be concatenated
        :param landscape: landscape object
        :return: numpy array with the concatenation of proposals based on rank
        """

        configuration = np.empty(self.n)

        for i in range(self.agents):
            for j in range(self.n):
                if self.allocation[i, j] == 1:
                    configuration[j] = self.agents_proposals[i][rank][j]

        return configuration

    def get_decision_making_mode(self):
        """
        Returns the decision making mode of the coordinator

        :return: the decision making mode of the coordinator
        """
        return self.decision_making_mode

    def receive_agents_offers(self, agent: int, offers: dict):
        """
        Receive the genes an agent wants to offer in a certain timestep in a dictionary with w entries, offer and threshold
        and stores this information in the `self.agents_offers` dictionary.

        Parameters:
        agent (int): The identifier of the agent making the offer.
        offers (dict): A dictionary containing the offers made by the agent. The keys are the genes being offered, and the values are the thresholds for the offers.

        Returns:
        None
        """
        self.reallocation[agent] = dict()
        self.reallocation[agent] = offers
        self.reallocation[agent]["bids"] = dict()

    def receive_agents_bids(self, agent: int, bids: dict):
        """
        Receive the bids of all agents and store them in the `reallocation` attribute.

        Args:
        - agent: the index of the agent sending the bids.
        - bids: a dictionary with the bids of all agents, with each key being an integer (the index of the agent) and each value being a dictionary containing the bid of that agent and its threshold.

        Returns:
        - None. The function updates the `reallocation` attribute of the object.
        """
        for i in range(self.agents):
            self.reallocation[i]["bids"][agent] = bids[i]["bid"]
            # print(self.reallocation)

    def get_agents_offers(self) -> dict:
        """
        Returns the offers made by agents in the current reallocation round.
        :return: a dictionary of dictionaries, where each key is an agent and the corresponding value is a dictionary with the offers made by that agent.
        """
        return self.reallocation

    def update_allocation(self) -> dict:
        """
        after receiving offers from agents (receive_agents_offer) and bids (receive_agents_bids) the coordinator computes
        the task re-allocation following the idea of a second price auction
        :return: dict with reallocation pattern. Swaps is computed and returned just in case the user wants to store it later on!
        """

        rng = np.random.default_rng()

        # initialize datastructure to store the reallocation
        swaps = dict()
        # print(self.reallocation)
        # loop over agents that made an offer
        for i in range(self.agents):
            swaps[i] = dict()
            swaps[i]["reallocate"] = False
            swaps[i]["gene"] = self.reallocation[i]["gene"]
            threshold = self.reallocation[i]["threshold"]
            # loop over all agents that submitted a bid
            for j in range(self.agents):
                if i != j:
                    if self.reallocation[i]["bids"][j] > threshold:
                        threshold = self.reallocation[i]["bids"][j]
                        swaps[i]["reallocate"] = True
                        swaps[i]["to_agent"] = j
                    # for the case that multiple agents submitted the same bid
                    # allocate between the two agents with uniform probability
                    elif (
                        swaps[i]["reallocate"] == True
                        and self.reallocation[i]["bids"][j] == threshold
                    ):
                        first_agent = swaps[i]["to_agent"]
                        second_agent = j
                        swaps[i]["to_agent"] = rng.choice([first_agent, second_agent])

        for i in range(self.agents):
            if swaps[i]["reallocate"] == True:
                self.allocation[i, swaps[i]["gene"]] = 0
                self.allocation[swaps[i]["to_agent"], swaps[i]["gene"]] = 1

        # print(f"Coordinator reallocates as follows:")
        # print(swaps)
        return swaps


def main():
    print("This is an extended version of Kauffman's NK model!")


if __name__ == "__main__":
    main()
