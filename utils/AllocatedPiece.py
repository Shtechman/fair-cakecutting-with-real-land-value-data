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

	def getCuts(self):
		return self.iFrom, self.iTo

	def getValue(self):
		"""
		The current agent evaluates his own piece.
		"""
		return self.agent.evaluationOfPiece(self)

	def getRelativeValue(self):
		"""
		The current agent evaluates his own piece relative to the entire cake.
		"""
		a = self.agent.evaluationOfPiece(self)
		b = self.agent.evaluationOfCake()
		c = a/b

		return c

	def getEnvy(self, otherPiece):
		"""
		The current agent reports his relative envy of the other agent's piece.
		"""
		enviousValue = self.agent.evaluationOfPiece(self)
		enviedValue  = self.agent.evaluationOfPiece(otherPiece);
		if enviousValue>=enviedValue:
			return 0
		else:
			return (enviedValue/enviousValue)-1

	def getLargestEnvy(self, otherPieces):
		"""
		The current agent reports his largest relative envy of another agent's piece.
		"""
		def getEnvy(piece):
			return self.getEnvy(piece)
		return max(list(map(lambda otherPiece: getEnvy(otherPiece), otherPieces)))


class AllocatedPiece2D:
	"""
	A class representing a piece allocated to an agent on a 2-dimensional cake.

	@author Itay Shtechman
	@since 2018-11
	"""

	def __init__(self, agent, iHorFrom=None, iVerFrom=None, iHorTo=None, iVerTo=None):
		""" /**
		 *  Initialize a 1-dimensional allocated piece based on a value function.
		 *  @param agent an Agent.
		 *  @param iHorFrom (float) start of Horizontal allocation.
		 *  @param iVerFrom (float) start of Vertical allocation.
		 *  @param iHorTo (float) end of Horizontal allocation.
		 *  @param iVerTo (float) end of Vertical allocation.
		 */ """
		if iHorFrom is None: iHorFrom = 0
		if iHorTo is None:   iHorTo = agent.valueFunction.getHorizontalDim()
		if iVerFrom is None: iVerFrom = 0
		if iVerTo is None:   iVerTo = agent.valueFunction.getVerticalDim()

		self.agent = agent
		self.iHorFrom = iHorFrom
		self.iHorTo = iHorTo
		self.iVerFrom = iVerFrom
		self.iVerTo = iVerTo

	def __repr__(self):
		return "%s receives [%0.2f,%0.2f,%0.2f,%0.2f]" % (self.agent.name, self.iHorFrom, self.iVerFrom, self.iHorTo, self.iVerTo)

	def getCuts(self):
		return self.iHorFrom, self.iVerFrom, self.iHorTo, self.iVerTo

	def getValue(self):
		"""
		The current agent evaluates his own piece.
		"""
		return self.agent.evaluationOfPiece(self)

	def getRelativeValue(self):
		"""
		The current agent evaluates his own piece relative to the entire cake.
		"""
		a = self.agent.evaluationOfPiece(self)
		b = self.agent.evaluationOfCake()
		c = a/b

		return c

	def getEnvy(self, otherPiece):
		"""
		The current agent reports his relative envy of the other agent's piece.
		"""
		enviousValue = self.agent.evaluationOfPiece(self)
		enviedValue  = self.agent.evaluationOfPiece(otherPiece);
		if enviousValue>=enviedValue:
			return 0
		else:
			return (enviedValue/enviousValue)-1

	def getLargestEnvy(self, otherPieces):
		"""
		The current agent reports his largest relative envy of another agent's piece.
		"""
		def getEnvy(piece):
			return self.getEnvy(piece)
		return max(list(map(lambda otherPiece: getEnvy(otherPiece), otherPieces)))
