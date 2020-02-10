#!python3

"""
/**
 * Implementation of the Assessor cake-cutting algorithm on a 2-dimensional cake.
 *
 */
"""

from functools import lru_cache
from math import log, ceil, floor

from utils.AllocatedPiece import AllocatedPiece
from utils.Cutter import SimpleCutter
from utils.Types import CutPattern


class AlgorithmAssessor:
    """ This Assessor uses a given cake-cutting algorithm and ignores the subjective valuations """
    def __init__(self, assessorAgentPool, assessorAlgorithm):
        self.assessorAgentPool = assessorAgentPool
        self.assessorAlgorithm = assessorAlgorithm

    def run(self, agents, cut_pattern):
        identicalPartitionWithIdenticalAgents = self._runAssessorAlgorithm(len(agents), cut_pattern)
        # Create virtual agents with the assessor's value function
        identicalPartitionWithDifferentAgents = list(map(
            lambda pair: AllocatedPiece(pair[0], pair[1].getIFromRow(), pair[1].getIFromCol(), pair[1].getIToRow(),
                                        pair[1].getIToCol()),
            zip(agents, identicalPartitionWithIdenticalAgents)
        ))
        return identicalPartitionWithDifferentAgents

    def getAlgorithmType(self):
        return "{}_{}".format("Assessor", self.assessorAlgorithm.getAlgorithmType())

    @lru_cache()
    def _runAssessorAlgorithm(self, numOfAgents, cut_pattern):
        agentsWithAssessorValueFunction = self.assessorAgentPool[:numOfAgents]
        # Run the assessor's division algorithm on the virtual agents:
        return self.assessorAlgorithm.run(agentsWithAssessorValueFunction, cut_pattern)


class AlgorithmSimpleAssessor:
    """ This Assessor uses a simple algorithm to divide the cake.
        The cake is divided into 2^ceil(log2(n)/2) horizontal slices and 2^floor(log2(n)/2) vertical slices """
    def __init__(self, assessorAgentPool):
        self.assessorAgentPool = assessorAgentPool

    def run(self, agents, cut_pattern):

        identicalPartitionWithIdenticalAgents = self._runAssessorAlgorithm(len(agents))
        # Create virtual agents with the assessor's value function
        identicalPartitionWithDifferentAgents = list(map(
            lambda pair: AllocatedPiece(pair[0], pair[1].getIFromRow(), pair[1].getIFromCol(), pair[1].getIToRow(),
                                        pair[1].getIToCol()),
            zip(agents, identicalPartitionWithIdenticalAgents)
        ))
        return identicalPartitionWithDifferentAgents

    def getAlgorithmType(self):
        return "{}_{}".format("Assessor", "NoPattern")

    @lru_cache()
    def _runAssessorAlgorithm(self, number_of_agents):
        # Run the assessor's division algorithm on the virtual agents:
        log_num_agents = log(float(number_of_agents), 2)
        num_hor_slices = pow(2, ceil(log_num_agents / 2.))
        num_ver_slices = pow(2, floor(log_num_agents / 2.))

        if num_hor_slices * num_ver_slices != number_of_agents:
            raise ValueError('Something went wrong in calculation.')  # test calculation logic

        horizontal_cutter = SimpleCutter(CutPattern.Hor)
        vertical_cutter = SimpleCutter(CutPattern.Ver)

        agents_for_hor_assessor_slices = self.assessorAgentPool[:num_hor_slices]
        initial_allocations = list(map(AllocatedPiece, agents_for_hor_assessor_slices))
        horizontal_slices = self._runRecursive(initial_allocations, horizontal_cutter)

        allocations = []
        for hor_slice in horizontal_slices:
            agents_for_ver_assessor_slices = self.assessorAgentPool[:num_ver_slices]
            hor_allocations = [hor_slice.getAllocatedPiece(agent) for agent in agents_for_ver_assessor_slices]
            allocations = allocations + self._runRecursive(hor_allocations, vertical_cutter)

        return allocations

    def _runRecursive(self, allocations, cutter):
        num_of_agents = len(allocations)
        if num_of_agents == 1:
            return allocations  # allocate the entire cake to the single agent.

        first_part_allocations, second_part_allocations = cutter.allocate_cuts(allocations, num_of_agents)

        return self._runRecursive(first_part_allocations, cutter.get_firt_part_cutter()) + \
               self._runRecursive(second_part_allocations, cutter.get_second_part_cutter())


if __name__ == '__main__':
    import doctest

    doctest.testmod()
