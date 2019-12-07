#!python3

"""
/**
 * Division of a 2-D cake using an Even Paz Algorithm
 *
 * @author Erel Segal-Halevi, Gabi Burabia, Itay Shtechman
 * @since 2016-11
 */
"""
from itertools import product

from utils.AllocatedPiece import AllocatedPiece
from utils.Cutter import EPCutter
from utils.Types import CutPattern, CutDirection


class AlgorithmEvenPaz:

    def run(self, agents, cut_pattern):
        """
        Calculate a proportional cake-division using the algorithm of Even and Paz (1984).
        @param agents - a list of n Agents, each with a value-function on the same cake.
        @return a list of n AllocatedPiece1D-s, each of which contains an Agent and an allocated part of the cake.
        todo: re-write all examples for 2d case
        >>> alg = AlgorithmEvenPaz()
        >>> Alice = Agent(name="Alice", valueFunction=ValueFunction1D([1,2,3,4]))
        >>> alg.run([Alice])
        [Alice receives [0.00,4.00]]

        >>> Bob = Agent(name="Bob", valueFunction=ValueFunction1D([40,30,20,10]))
        >>> alg.run([Alice,Bob])
        [Bob receives [0.00,2.00], Alice receives [2.00,4.00]]

        >>> Carl = Agent(name="Carl", valueFunction=ValueFunction1D([100,100,100,100]))
        >>> alg.run([Alice,Bob,Carl])
        [Bob receives [0.00,1.30], Carl receives [1.30,2.92], Alice receives [2.92,4.00]]
        """

        num_of_agents = len(agents)

        cutters = self._get_cutters_list(cut_pattern, num_of_agents)

        return self.aggregate_cutters(agents, cutters)

    def aggregate_cutters(self, agents, cutters):
        results = []
        for cutter in cutters:
            # initially, allocate the entire cake to all agents:
            initial_allocations = list(map(AllocatedPiece, agents))

            # now, recursively divide the cake among the agents using a given cutter:
            results.append((cutter, self._runRecursive(initial_allocations, cutter)))

        for r in results:
            print(r)
        if len(results) > 1:
            return {str(result[0]): result[1] for result in results}
        else:
            return results[0][1]

    def _get_cutters_list(self, cut_pattern, num_of_agents):
        if cut_pattern is CutPattern.BruteForce:
            cut_series_list = [list(i) for i in product([CutDirection.Horizontal,
                                                         CutDirection.Vertical], repeat=num_of_agents-1)]
            cutters = [EPCutter(cut_series) for cut_series in cut_series_list]
        else:
            cutters = [EPCutter(cut_pattern)]
        return cutters

    @staticmethod
    def getAlgorithmType():
        return "EvenPaz"

    def _runRecursive(self, allocations, cutter):
        num_of_agents = len(allocations)
        if num_of_agents == 1:
            return allocations  # allocate the entire cake to the single agent.

        first_part_allocations, second_part_allocations = cutter.allocate_cuts(allocations, num_of_agents)

        return self._runRecursive(first_part_allocations, cutter.get_firt_part_cutter()) +\
               self._runRecursive(second_part_allocations, cutter.get_second_part_cutter())


if __name__ == '__main__':
    from utils.ValueFunction1D import ValueFunction1D
    from utils.Agent import Agent


    import doctest
    # doctest.testmod()

    # demo test
    alg = AlgorithmEvenPaz()
    Alice = Agent(name="Alice", valueFunction=ValueFunction1D([1, 2, 3, 4]))
    Bob = Agent(name="Bob", valueFunction=ValueFunction1D([40, 30, 20, 10]))
    Carl = Agent(name="Carl", valueFunction=ValueFunction1D([100, 100, 100, 100]))
    print("when Alice is the only agent -", alg.run([Alice]))
    print("when Alice and Bob are the only agents -", alg.run([Alice, Bob]))
    print("when Carl and Bob are the only agents -", alg.run([Carl, Bob]))
    print("when Alice, Bob and Carl are the agents -", alg.run([Alice, Bob, Carl]))
