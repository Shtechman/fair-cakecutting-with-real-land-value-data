#!python3

"""
/**
 * Implementation of the Even-Paz proportional cake-cutting algorithm on a 1-dimensional cake.
 *
 * @author Erel Segal-Halevi, Gabi Burabia, Itay Shtechman
 * @since 2016-11
 */
"""

from functools import lru_cache
from utils.ValueFunction1D import ValueFunction1D
from utils.Agent import Agent
from utils.AllocatedPiece import AllocatedPiece1D, AllocatedPiece
from utils.AlgorithmEvenPaz import AlgorithmEvenPaz


class AlgorithmAssessor:

	def __init__(self, assessorAgentPool, assessorAlgorithm):
		"""
		@param assessorValuationFunction a ValueFunction1D that represents the valuation according to which the assessor divides the land.
		   """
		self.assessorAgentPool = assessorAgentPool
		self.assessorAlgorithm = assessorAlgorithm

	def run(self, agents, cut_pattern):
		identicalPartitionWithIdenticalAgents = self._runAssessorAlgorithm(len(agents), cut_pattern)
		# Create virtual agents with the assessor's value function
		identicalPartitionWithDifferentAgents = list(map(
			lambda pair: AllocatedPiece(pair[0], pair[1].getIFromRow(), pair[1].getIFromCol(), pair[1].getIToRow(), pair[1].getIToCol()),
			zip(agents, identicalPartitionWithIdenticalAgents)
			))
		return identicalPartitionWithDifferentAgents

	@staticmethod
	def getAlgorithmType():
		return "Assessor"

	@lru_cache()
	def _runAssessorAlgorithm(self, numOfAgents, cut_pattern):
		agentsWithAssessorValueFunction = self.assessorAgentPool[:numOfAgents]
		# Run the assessor's division algorithm on the virtual agents:
		return self.assessorAlgorithm.run(agentsWithAssessorValueFunction, cut_pattern)

if __name__ == '__main__':


	import doctest
	doctest.testmod()
