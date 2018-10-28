#!python3

"""
/**
 * Implementation of the Even-Paz proportional cake-cutting algorithm on a 1-dimensional cake.
 *
 * @author Erel Segal-Halevi, Gabi Burabia
 * @since 2016-11
 */
"""

from functools import lru_cache
import numpy as np
import operator

class AlgorithmAssessor1D:

	def __init__(self, assessorValuationFunction, assessorAlgorithm):
		"""
		@param assessorValuationFunction a ValueFunction1D that represents the valuation according to which the assessor divides the land.
		   """
		self.assessorValuationFunction = assessorValuationFunction
		self.assessorAlgorithm = assessorAlgorithm

	def run(self, agents):
		"""
		Calculate a proportional cake-division based on assessorValuationFunction.
		@param agents - a list of n Agents, each with a value-function on the same cake.
		@return a list of n AllocatedPiece1D-s, each of which contains an Agent and an allocated part of the cake.

		>>> alg = AlgorithmAssessor1D(ValueFunction1D([1,1,2,4]), AlgorithmEvenPaz1D())
		>>> Alice = Agent(name="Alice", valueFunction=ValueFunction1D([1,2,3,4]))
		>>> alg.run([Alice])
		[Alice receives [0.00,4.00]]

		>>> Bob = Agent(name="Bob", valueFunction=ValueFunction1D([40,30,20,10]))
		>>> alg.run([Alice,Bob])
		[Alice receives [0.00,3.00], Bob receives [3.00,4.00]]

		>>> Carl = Agent(name="Carl", valueFunction=ValueFunction1D([100,100,100,100]))
		>>> alg.run([Alice,Bob,Carl])
		[Alice receives [0.00,2.33], Bob receives [2.33,3.33], Carl receives [3.33,4.00]]
		"""

		identicalPartitionWithIdenticalAgents = self._runAssessorAlgorithm(len(agents))
		# Create virtual agents with the assessor's value function
		identicalPartitionWithDifferentAgents = map(
			lambda pair: AllocatedPiece1D(pair[0], pair[1].iFrom, pair[1].iTo),
			zip(agents, identicalPartitionWithIdenticalAgents)
			)
		return list(identicalPartitionWithDifferentAgents)

	@lru_cache()
	def _runAssessorAlgorithm(self, numOfAgents):
		agentsWithAssessorValueFunction = map(Agent, self.assessorValuationFunction.noisyValuesArray(0, None, numOfAgents));
		# Run the assessor's division algorithm on the virtual agents:
		return self.assessorAlgorithm.run(agentsWithAssessorValueFunction)

if __name__ == '__main__':
	from ValueFunction1D import ValueFunction1D
	from Agent import Agent
	from AllocatedPiece1D import AllocatedPiece1D
	from AlgorithmEvenPaz1D import AlgorithmEvenPaz1D

	import doctest
	doctest.testmod()
