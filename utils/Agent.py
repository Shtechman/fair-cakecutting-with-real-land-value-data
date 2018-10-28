#!python3

from functools import lru_cache

class Agent:
	"""
	an agent has a name and a value-function.
	"""
	def __init__(self, valueFunction, name="Anonymous"):
		self.name = name
		self.valueFunction = valueFunction

	@lru_cache()
	def evalQuery(self, iFrom, iTo):
		return self.valueFunction.sum(iFrom, iTo)

	@lru_cache()
	def markQuery(self, iFrom, value):
		return self.valueFunction.invSum(iFrom, value)

	@lru_cache()
	def evaluationOfPiece(self, piece):
		return self.evalQuery(piece.iFrom, piece.iTo)

	@lru_cache()
	def evaluationOfCake(self):
		return self.valueFunction.getValueOfEntireCake()
