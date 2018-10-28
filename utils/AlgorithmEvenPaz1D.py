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

from .AllocatedPiece1D import AllocatedPiece1D
### with line - does not work from main
### without line - does not work from AlgorithmAssessor1D

class AlgorithmEvenPaz1D:

	def run(self, agents):
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
		# initially, allocate the entire cake to all agents:
		initialAllocations = list(map(AllocatedPiece1D, agents))
		# now, recursively divide the cake among the agents:
		return self._runRecursive(initialAllocations)

	def _runRecursive(self, allocations):
		numOfAgents = len(allocations)
		if numOfAgents==1:
			return allocations  # allocate the entire cake to the single agent.
		numOfAgentsInFirstPartition = int(np.ceil(numOfAgents/2))
		proportionOfFirstPartition = numOfAgentsInFirstPartition / float(numOfAgents)

		# Ask all agents a "cut" query - cut the cake in proportionOfFirstPartition (half or near-half):
		for allocation in allocations:
			allocation.halfCut = allocation.agent.markQuery(allocation.iFrom, proportionOfFirstPartition*allocation.getValue())

		# Calculate the median of the agents' half-cuts: this will be our cut location.
		allocations.sort(key=operator.attrgetter('halfCut'))
		endOfFirstPart = allocations[numOfAgentsInFirstPartition-1].halfCut;
		startOfSecondPart = allocations[numOfAgentsInFirstPartition].halfCut;
		cutLocation = (endOfFirstPart+startOfSecondPart)/2;

		# Divide the agents to two groups of nearly the same size, based on their half-cut locations:
		firstPartAllocations = []
		secondPartAllocations = []
		for i in range(0, numOfAgentsInFirstPartition):
			firstPartAllocations.append(
				AllocatedPiece1D(allocations[i].agent, allocations[i].iFrom, cutLocation));
		for i in range(numOfAgentsInFirstPartition,  numOfAgents):
			secondPartAllocations.append(
				AllocatedPiece1D(allocations[i].agent, cutLocation,   allocations[i].iTo));
		return self._runRecursive(firstPartAllocations) + self._runRecursive(secondPartAllocations);


if __name__ == '__main__':
	from ValueFunction1D import ValueFunction1D
	from Agent import Agent
	from AllocatedPiece1D import AllocatedPiece1D


	import doctest
	doctest.testmod()
