#!python3

"""
/**
 * Implementation of the Even-Paz proportional cake-cutting algorithm on a 1-dimensional cake.
 *
 * @author Erel Segal-Halevi, Gabi Burabia
 * @since 2016-11
 */
"""
from ValueFunction1D import ValueFunction1D
from Agent import Agent
from AllocatedPiece1D import AllocatedPiece1D

import operator
import numpy as np


def proportionalDivisionEvenPaz(agents):
	"""
	Calculate a proportional cake-division using the algorithm of Even and Paz (1984).
	@param agents - a list of Agents.
	@return the same value functions, such that each value function is on a distinct part of the cake.

	>>> Alice = Agent(name="Alice", valueFunction=ValueFunction1D([1,2,3,4]))
	>>> proportionalDivisionEvenPaz([Alice])
	[Alice receives [0.00,4.00]]

	>>> Bob = Agent(name="Bob", valueFunction=ValueFunction1D([40,30,20,10]))
	>>> proportionalDivisionEvenPaz([Alice,Bob])
	[Bob receives [0.00,2.00], Alice receives [2.00,4.00]]

	>>> Carl = Agent(name="Carl", valueFunction=ValueFunction1D([100,100,100,100]))
	>>> proportionalDivisionEvenPaz([Alice,Bob,Carl])
	[Bob receives [0.00,1.30], Carl receives [1.30,2.92], Alice receives [2.92,4.00]]
	"""
	# initially, allocate the entire cake to all agents:
	initialAllocations = list(map(AllocatedPiece1D, agents))
	# now, recursively divide the cake among the agents:
	return proportionalDivisionEvenPazRecursive(initialAllocations)

def proportionalDivisionEvenPazRecursive(allocations):
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
	return proportionalDivisionEvenPazRecursive(firstPartAllocations) + proportionalDivisionEvenPazRecursive(secondPartAllocations);


if __name__ == '__main__':
	import doctest
	doctest.testmod()
