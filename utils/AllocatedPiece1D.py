#!python3

from functools import lru_cache

class AllocatedPiece1D:
	"""
	A class representing a piece allocated to an agent on a 1-dimensional cake.

	@author Erel Segal-Halevi
	@since 2016-11
	"""

	def __init__(self, agent, iFrom=None, iTo=None):
		""" /**
		 *  Initialize a 1-dimensional allocated piece based on a value function.
		 *  @param agent an Agent.
		 *  @param iFrom (float) start of allocation.
		 *  @param iTo (float) end of allocation.
		 */ """
		if iFrom is None: iFrom = 0
		if iTo is None:   iTo = agent.valueFunction.length

		self.agent = agent
		self.iFrom = iFrom
		self.iTo = iTo

	def __repr__(self):
		return "%s receives [%0.2f,%0.2f]" % (self.agent.name, self.iFrom, self.iTo)

	def getValue(self):
		"""
		The current agent evaluates his own piece.
		"""
		return self.agent.evaluationOfPiece(self)

	def getRelativeValue(self):
		"""
		The current agent evaluates his own piece relative to the entire cake.
		"""
		return self.agent.evaluationOfPiece(self) / self.agent.evaluationOfCake()

	def getEnvy(self, otherPiece):
		"""
		The current agent reports his relative envy of the other agent's piece.
		"""
		enviousValue = self.agent.evaluationOfPiece(self)
		enviedValue  = self.agent.evaluationOfPiece(otherPiece);
		if enviousValue>=enviedValue:
			return 0
		else:
			return (enviedValue-enviousValue)/enviousValue

	def getLargestEnvy(self, otherPieces):
		"""
		The current agent reports his largest relative envy of another agent's piece.
		"""
		def getEnvy(piece):
			return self.getEnvy(piece)
		maxEnviedPiece = max(otherPieces, getEnvy)
		return self.getEnvy(maxEnviedPiece)
