#!python3

"""
/**
 * Division of a 1-D cake using an "objective" assessor.
 *
 * @author Erel Segal-Halevi, Gabi Burabia
 * @since 2016-11
 */
"""
import os

import numpy as np
import operator
from copy import copy
from utils.AllocatedPiece import AllocatedPiece1D, AllocatedPiece


### with line - does not work from main
### without line - does not work from AlgorithmAssessor1D
from utils.Cutter import Cutter


class AlgorithmEvenPaz:

    def run(self, agents, cut_pattern):
        """
        Calculate a proportional cake-division using the algorithm of Even and Paz (1984).
        @param agents - a list of n Agents, each with a value-function on the same cake.
        @return a list of n AllocatedPiece1D-s, each of which contains an Agent and an allocated part of the cake.

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
        # initially, allocate the entire cake to all agents:
        initial_allocations = list(map(AllocatedPiece, agents))
        # now, recursively divide the cake among the agents:
        return self._runRecursive(initial_allocations, Cutter(cut_pattern))

    @staticmethod
    def getAlgorithmType():
        return "EvenPaz"

    def _runRecursive(self, allocations, cutter):
        num_of_agents = len(allocations)
        if num_of_agents == 1:
            return allocations  # allocate the entire cake to the single agent.

        first_part_allocations, second_part_allocations = cutter.allocate_cuts(allocations, num_of_agents)

        return self._runRecursive(first_part_allocations, copy(cutter)) +\
               self._runRecursive(second_part_allocations, copy(cutter))


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
