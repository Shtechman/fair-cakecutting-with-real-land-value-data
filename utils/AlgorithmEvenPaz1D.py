#!python3

"""
/**
 * Division of a 1-D cake using an "objective" assessor.
 *
 * @author Erel Segal-Halevi, Gabi Burabia
 * @since 2016-11
 */
"""
import numpy as np
import operator
import copy
from utils.AllocatedPiece import AllocatedPiece1D, AllocatedPiece


### with line - does not work from main
### without line - does not work from AlgorithmAssessor1D

class AlgorithmEvenPaz1D:

    def run(self, agents, cutDirections):
        """
        Calculate a proportional cake-division using the algorithm of Even and Paz (1984).
        @param agents - a list of n Agents, each with a value-function on the same cake.
        @return a list of n AllocatedPiece1D-s, each of which contains an Agent and an allocated part of the cake.

        >>> alg = AlgorithmEvenPaz1D()
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
        self.num_of_directions = len(cutDirections)
        self.cut_directions = cutDirections
        current_cut_index = 0
        # initially, allocate the entire cake to all agents:
        initialAllocations = list(map(AllocatedPiece, agents))
        # now, recursively divide the cake among the agents:
        return self._runRecursive(initialAllocations, current_cut_index)

    @staticmethod
    def getAlgorithmType():
        return "EvenPaz"

    def _nextCutDirectionIndex(self, current_cut_index):
        return (current_cut_index+1) % self.num_of_directions

    def _runRecursive(self, allocations, current_cut_index):
        cutDirection = self.cut_directions[current_cut_index]
        numOfAgents = len(allocations)
        if numOfAgents==1:
            return allocations  # allocate the entire cake to the single agent.
        numOfAgentsInFirstPartition = int(np.ceil(numOfAgents/2))
        proportionOfFirstPartition = numOfAgentsInFirstPartition / float(numOfAgents)

        # Ask all agents a "cut" query - cut the cake in proportionOfFirstPartition (half or near-half):
        for allocation in allocations:
            allocation.halfCut = allocation.markQuery(proportionOfFirstPartition*allocation.getValue(), cutDirection)

        # Calculate the median of the agents' half-cuts: this will be our cut location.
        allocations.sort(key=operator.attrgetter('halfCut'))
        endOfFirstPart = allocations[numOfAgentsInFirstPartition-1].halfCut
        startOfSecondPart = allocations[numOfAgentsInFirstPartition].halfCut
        cutLocation = (endOfFirstPart+startOfSecondPart)/2

        # Divide the agents to two groups of nearly the same size, based on their half-cut locations:
        firstPartAllocations = []
        secondPartAllocations = []

        for i in range(0, numOfAgentsInFirstPartition):
            iFrom = allocations[i].getDirectionaliFrom(cutDirection)
            iTo = cutLocation
            firstPartAllocations.append(allocations[i].subCut(iFrom, iTo, cutDirection))

        for i in range(numOfAgentsInFirstPartition,  numOfAgents):
            iFrom = cutLocation
            iTo = allocations[i].getDirectionaliTo(cutDirection)
            secondPartAllocations.append(allocations[i].subCut(iFrom, iTo, cutDirection))

        next_cut_index = self._nextCutDirectionIndex(current_cut_index)
        return self._runRecursive(firstPartAllocations, next_cut_index) +\
               self._runRecursive(secondPartAllocations, next_cut_index)


if __name__ == '__main__':
    from utils.ValueFunction1D import ValueFunction1D
    from utils.Agent import Agent


    import doctest
    # doctest.testmod()

    # demo test
    alg = AlgorithmEvenPaz1D()
    Alice = Agent(name="Alice", valueFunction=ValueFunction1D([1, 2, 3, 4]))
    Bob = Agent(name="Bob", valueFunction=ValueFunction1D([40, 30, 20, 10]))
    Carl = Agent(name="Carl", valueFunction=ValueFunction1D([100, 100, 100, 100]))
    print("when Alice is the only agent -", alg.run([Alice]))
    print("when Alice and Bob are the only agents -", alg.run([Alice, Bob]))
    print("when Carl and Bob are the only agents -", alg.run([Carl, Bob]))
    print("when Alice, Bob and Carl are the agents -", alg.run([Alice, Bob, Carl]))
