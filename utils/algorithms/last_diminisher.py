#!python3

"""
/**
 * Division of a 2-D cake using a Last Diminisher Algorithm
 *
 * @author Itay Shtechman
 * @since 2019-11
 */
"""
from itertools import product

from utils.simulation.allocated_piece import AllocatedPiece
from utils.cutters.last_diminisher_cutter import LDCutter
from utils.simulation.cc_types import CutPattern, CutDirection


# todo: this class can be refactored because it is very similar to even_paz.py
class AlgorithmLastDiminisher:
    def run(self, agents, cut_pattern):

        num_of_agents = len(agents)
        cutters = self._get_cutters_list(cut_pattern, num_of_agents)

        return self.aggregate_cutters(agents, cutters)

    @staticmethod
    def _get_cutters_list(cut_pattern, num_of_agents):
        if cut_pattern is CutPattern.BruteForce:
            cut_series_list = [
                list(i)
                for i in product(
                    [CutDirection.Horizontal, CutDirection.Vertical],
                    repeat=num_of_agents - 1,
                )
            ]
            cutters = [LDCutter(cut_series) for cut_series in cut_series_list]
        else:
            cutters = [LDCutter(cut_pattern)]
        return cutters

    @staticmethod
    def get_algorithm_type():
        return "LastDiminisher"

    def aggregate_cutters(self, agents, cutters):
        results = []
        for cutter in cutters:
            """ Initially, allocate the entire cake to all agents """
            initial_allocations = list(map(AllocatedPiece, agents))

            """ Now, recursively divide the cake among the agents using a given cutter """
            results.append(
                (cutter, self._run_recursive(initial_allocations, cutter))
            )

        if len(results) > 1:
            return {str(result[0]): result[1] for result in results}
        else:
            return results[0][1]

    def _run_recursive(self, allocations, cutter):
        num_of_agents = len(allocations)
        if num_of_agents == 1:
            """ Allocate the entire cake to the single agent """
            return allocations

        first_part_allocations, second_part_allocations = cutter.allocate_cuts(
            allocations, num_of_agents
        )

        return self._run_recursive(
            first_part_allocations, cutter.get_first_part_cutter()
        ) + self._run_recursive(
            second_part_allocations, cutter.get_second_part_cutter()
        )


if __name__ == "__main__":
    pass
