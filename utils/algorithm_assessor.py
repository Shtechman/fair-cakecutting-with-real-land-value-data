#!python3

"""
/**
 * Implementation of the Assessor cake-cutting algorithm on a 2-dimensional cake.
 *
 * @author Itay Shtechman
 * @since 2019-11
 */
"""

from functools import lru_cache
from math import log, ceil, floor

from utils.allocated_piece import AllocatedPiece
from utils.cutter import SimpleCutter
from utils.types import CutPattern


class AlgorithmAssessor:
    """ This Assessor uses a given cake-cutting algorithm and ignores the subjective valuations """

    def __init__(self, assessor_agent_pool, assessor_cutting_algorithm):
        self.assessor_agent_pool = assessor_agent_pool
        self.assessor_cutting_algorithm = assessor_cutting_algorithm

    def run(self, agents, cut_pattern):
        identical_partition_with_identical_agents = self._run_assessor_algorithm(
            len(agents), cut_pattern
        )

        """ Create virtual agents with the assessor's value function """
        identical_partition_with_different_agents = list(
            map(
                lambda pair: AllocatedPiece(
                    pair[0],
                    pair[1].get_i_from_row(),
                    pair[1].get_i_from_col(),
                    pair[1].get_i_to_row(),
                    pair[1].get_i_toCol(),
                ),
                zip(agents, identical_partition_with_identical_agents),
            )
        )
        return identical_partition_with_different_agents

    def get_algorithm_type(self):
        return "{}_{}".format(
            "Assessor", self.assessor_cutting_algorithm.get_algorithm_type()
        )

    @lru_cache()
    def _run_assessor_algorithm(self, num_of_agents, cut_pattern):
        agents_with_assessor_value_function = self.assessor_agent_pool[
            :num_of_agents
        ]

        """ Run the assessor's division algorithm on the virtual agents:"""
        return self.assessor_cutting_algorithm.run(
            agents_with_assessor_value_function, cut_pattern
        )


class AlgorithmSimpleAssessor:
    """ This Assessor uses a simple algorithm to divide the cake.
        The cake is divided into 2^ceil(log2(n)/2) horizontal slices and 2^floor(log2(n)/2) vertical slices """

    def __init__(self, assessor_agent_pool):
        self.assessor_agent_pool = assessor_agent_pool

    def run(self, agents, cut_pattern):

        identical_partition_with_identical_agents = self._run_assessor_algorithm(
            len(agents)
        )

        """ Create virtual agents with the assessor's value function """
        identical_partition_with_different_agents = list(
            map(
                lambda pair: AllocatedPiece(
                    pair[0],
                    pair[1].get_i_from_row(),
                    pair[1].get_i_from_col(),
                    pair[1].get_i_to_row(),
                    pair[1].get_i_to_col(),
                ),
                zip(agents, identical_partition_with_identical_agents),
            )
        )
        return identical_partition_with_different_agents

    def get_algorithm_type(self):
        return "{}_{}".format("Assessor", "NoPattern")

    @lru_cache()
    def _run_assessor_algorithm(self, num_of_agents):
        log_num_agents = log(float(num_of_agents), 2)
        num_hor_slices = pow(2, ceil(log_num_agents / 2.0))
        num_ver_slices = pow(2, floor(log_num_agents / 2.0))

        if num_hor_slices * num_ver_slices != num_of_agents:
            raise ValueError("Something went wrong in calculation.")

        horizontal_cutter = SimpleCutter(CutPattern.Hor)
        vertical_cutter = SimpleCutter(CutPattern.Ver)

        agents_for_hor_assessor_slices = self.assessor_agent_pool[
            :num_hor_slices
        ]
        initial_allocations = list(
            map(AllocatedPiece, agents_for_hor_assessor_slices)
        )
        horizontal_slices = self._run_recursive(
            initial_allocations, horizontal_cutter
        )

        allocations = []
        for hor_slice in horizontal_slices:
            agents_for_ver_assessor_slices = self.assessor_agent_pool[
                :num_ver_slices
            ]
            hor_allocations = [
                hor_slice.get_allocated_piece(agent)
                for agent in agents_for_ver_assessor_slices
            ]
            allocations = allocations + self._run_recursive(
                hor_allocations, vertical_cutter
            )

        return allocations

    def _run_recursive(self, allocations, cutter):
        num_of_agents = len(allocations)
        if num_of_agents == 1:
            return allocations  # allocate the entire cake to the single agent.

        first_part_allocations, second_part_allocations = cutter.allocate_cuts(
            allocations, num_of_agents
        )

        return self._run_recursive(
            first_part_allocations, cutter.get_first_part_cutter()
        ) + self._run_recursive(
            second_part_allocations, cutter.get_second_part_cutter()
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
