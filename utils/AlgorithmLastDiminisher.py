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

from utils.AllocatedPiece import AllocatedPiece
from utils.Cutter import LDCutter
from utils.Types import CutPattern, CutDirection


# todo: this class can be refactored because it is very similar to AlgorithmEvenPaz.py
class AlgorithmLastDiminisher:

    def run(self, agents, cut_pattern):
        """
        Calculate a proportional cake-division using the algorithm of Last Diminisher (1948).
        @param agents - a list of n Agents, each with a value-function on the same cake.
        @return a list of n AllocatedPiece-s, each of which contains an Agent and an allocated part of the cake.
        todo: write examples for 2d case
        """

        num_of_agents = len(agents)

        cutters = self._get_cutters_list(cut_pattern, num_of_agents)

        return self.aggregate_cutters(agents, cutters)

    def _get_cutters_list(self, cut_pattern, num_of_agents):
        if cut_pattern is CutPattern.BruteForce:
            cut_series_list = [list(i) for i in product([CutDirection.Horizontal,
                                                         CutDirection.Vertical], repeat=num_of_agents-1)]
            cutters = [LDCutter(cut_series) for cut_series in cut_series_list]
        else:
            cutters = [LDCutter(cut_pattern)]
        return cutters

    @staticmethod
    def getAlgorithmType():
        return "LastDiminisher"

    def aggregate_cutters(self, agents, cutters):
        results = []
        for cutter in cutters:
            # initially, allocate the entire cake to all agents:
            initial_allocations = list(map(AllocatedPiece, agents))

            # now, recursively divide the cake among the agents using a given cutter:
            results.append((cutter, self._runRecursive(initial_allocations, cutter)))

        # for r in results:
        #     print(r)
        if len(results) > 1:
            return {str(result[0]): result[1] for result in results}
        else:
            return results[0][1]

    def _runRecursive(self, allocations, cutter):
        num_of_agents = len(allocations)
        if num_of_agents == 1:
            return allocations  # allocate the entire cake to the single agent.

        first_part_allocations, second_part_allocations = cutter.allocate_cuts(allocations, num_of_agents)

        return self._runRecursive(first_part_allocations, cutter.get_firt_part_cutter()) +\
               self._runRecursive(second_part_allocations, cutter.get_second_part_cutter())


if __name__ == '__main__':
    pass
