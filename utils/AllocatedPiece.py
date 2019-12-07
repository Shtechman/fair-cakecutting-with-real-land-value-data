#!python3

from functools import lru_cache

from utils.Types import CutDirection


class Piece:
	def __init__(self, iFromRow, iFromCol, iToRow, iToCol):
		""" /**
         *  Initialize a 2-dimensional piece.
         *  @param iFromRow float location of the start vertical cut
         *  @param iFromCol float location of the start horizontal cut
         *  @param iToRow float location of the end vertical cut
         *  @param iToCol float location of the end horizontal cut
         */ """
		self.iFromRow = iFromRow
		self.iFromCol = iFromCol
		self.iToRow = iToRow
		self.iToCol = iToCol

	def __repr__(self):
		return "[%0.2f,%0.2f,%0.2f,%0.2f] - (Ratio %0.2f)" % (self.iFromRow, self.iFromCol, self.iToRow, self.iToCol,
															  self.getFaceRatio())

	def toString(self):
		return self.__repr__()

	def getAllocatedPiece(self, agent):
		return AllocatedPiece(agent, self.iFromRow, self.iFromCol, self.iToRow, self.iToCol)

	def getOppositeDirectionalRange(self, direction):
		switcher = {
			CutDirection.Horizontal: (self.iFromCol, self.iToCol),
			CutDirection.Vertical: (self.iFromRow, self.iToRow)
		}

		return switcher.get(direction, None)

	def getDirectionaliFrom(self, direction):
		switcher = {
			CutDirection.Horizontal: self.iFromRow,
			CutDirection.Vertical: self.iFromCol,
		}

		return switcher.get(direction, None)

	def getDirectionaliTo(self, direction):
		switcher = {
			CutDirection.Horizontal: self.iToRow,
			CutDirection.Vertical: self.iToCol,
		}

		return switcher.get(direction, None)

	def getDimensions(self):
		return {CutDirection.Horizontal: self.iToRow - self.iFromRow,
				CutDirection.Vertical: self.iToCol - self.iFromCol}

	def getIFromCol(self):
		return self.iFromCol

	def getIToCol(self):
		return self.iToCol

	def getIFromRow(self):
		return self.iFromRow

	def getIToRow(self):
		return self.iToRow

	def getCuts(self):
		return self.iFromRow, self.iFromCol, self.iToRow, self.iToCol

	def getFaceRatio(self):
		width = self.iToCol - self.iFromCol
		height = self.iToRow - self.iFromRow
		return min(width, height) / max(width, height)


class AllocatedPiece(Piece):
	"""
	A class representing a piece allocated to an agent on a 2-dimensional cake.

	@author Itay Shtechman
	@since 2018-11
	"""

	def __init__(self, agent, iFromRow=None, iFromCol=None, iToRow=None, iToCol=None):
		""" /**
		 *  Initialize a 2-dimensional allocated piece based on a value function.
		 *  @param agent an Agent.
		 *  @param iFromRow float location of the start vertical cut
		 *  @param iFromCol float location of the start horizontal cut
		 *  @param iToRow float location of the end vertical cut
		 *  @param iToCol float location of the end horizontal cut
		 */ """
		if iFromRow is None: iFromRow = 0
		if iFromCol is None: iFromCol = 0
		if iToRow is None:   iToRow = agent.valueMapRows
		if iToCol is None:   iToCol = agent.valueMapCols

		super(AllocatedPiece, self).__init__(iFromRow, iFromCol, iToRow, iToCol)
		self.agent = agent
		self.agent_name = agent.name
		self.cutmarks = {}

	def __repr__(self):
		return "%s(%s) receives %s" % (self.agent_name, self.agent.file_num, super(AllocatedPiece, self).__repr__())

	def toString(self):
		return self.__repr__()

	def clear(self):
		self.agent.cleanMemory()
		del self.agent

	def subCut(self, iDirFrom, iDirTo, direction):
		switcher = {
			CutDirection.Horizontal: (iDirFrom, self.iFromCol, iDirTo, self.iToCol),
			CutDirection.Vertical: (self.iFromRow, iDirFrom, self.iToRow, iDirTo)
		}

		ihf, ivf, iht, ivt = switcher.get(direction, None)

		return AllocatedPiece(self.agent, ihf, ivf, iht, ivt)

	def getAgent(self):
		return self.agent

	def getDirectionalValue(self, cut_location, direction):
		if self.cutmarks[direction] < cut_location:
			subcut_iFrom = self.getDirectionaliFrom(direction)
			subcut_iTo = cut_location
		else:
			subcut_iFrom = cut_location
			subcut_iTo = self.getDirectionaliTo(direction)
		return self.subCut(subcut_iFrom, subcut_iTo, direction).getValue()

	def getDirectionalFaceRatio(self, cut_location, direction):
		if self.cutmarks[direction] < cut_location:
			subcut_iFrom = self.getDirectionaliFrom(direction)
			subcut_iTo = cut_location
		else:
			subcut_iFrom = cut_location
			subcut_iTo = self.getDirectionaliTo(direction)
		return self.subCut(subcut_iFrom, subcut_iTo, direction).getFaceRatio()

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
		enviedValue  = self.agent.evaluationOfPiece(otherPiece)
		if enviousValue >= enviedValue:
			return 1
		else:
			return (enviedValue / enviousValue)

	def getLargestEnvy(self, otherPieces):
		"""
		The current agent reports his largest relative envy of another agent's piece.
		"""
		def getEnvy(piece):
			return self.getEnvy(piece)
		return max(list(map(lambda otherPiece: getEnvy(otherPiece), otherPieces)))

	@lru_cache()
	def markQuery(self, value, cutDirection):
		switcher = {
			CutDirection.Horizontal: self.markQueryHorizontal,
			CutDirection.Vertical: self.markQueryVertical,
		}

		def errorFunc():
			raise ValueError("invalid direction: " + str(cutDirection))

		markQueryFunc = switcher.get(cutDirection, errorFunc)
		return markQueryFunc(value)

	@lru_cache()
	def markQueryHorizontal(self, value):
		return self.agent.markQuery(self.iFromRow, self.iFromCol, self.iToCol, value, CutDirection.Horizontal)

	@lru_cache()
	def markQueryVertical(self, value):
		return self.agent.markQuery(self.iFromCol, self.iFromRow, self.iToRow, value, CutDirection.Vertical)




#
# class AllocatedPiece1D:
# 	"""
#     A class representing a piece allocated to an agent on a 1-dimensional cake.
#
#     @author Itay Shtechman
#     @since 2018-11
#     """
#
# 	def __init__(self, agent, iFromCol=None, iToCol=None):
# 		""" /**
#          *  Initialize a 1-dimensional allocated piece based on a value function.
#          *  @param agent an Agent.
#          *  @param iFromCol float location of the start horizontal cut
#          *  @param iToCol float location of the end horizontal cut
#          */ """
# 		if iFromCol is None:
# 			iFromCol = 0
# 		if iToCol is None:
# 			iToCol = agent.valueMapCols
#
# 		self.piece2d = AllocatedPiece(agent, 0, iFromCol, agent.valueMapRows, iToCol)
#
# 	def __repr__(self):
# 		return "%s receives [%0.2f,%0.2f]" % (self.agent.name, self.piece2d.iFromCol, self.piece2d.iToCol)
#
# 	def getAgent(self):
# 		return self.piece2d.getAgent()
#
# 	def getIFrom(self):
# 		return self.piece2d.getDirectionaliFrom(CutDirection.Vertical)
#
# 	def getITo(self):
# 		return self.piece2d.getDirectionaliTo(CutDirection.Vertical)
#
# 	def getCuts(self):
# 		return self.piece2d.getCuts()
#
# 	def getValue(self):
# 		"""
#         The current agent evaluates his own piece.
#         """
# 		return self.piece2d.getValue()
#
# 	def getRelativeValue(self):
# 		"""
#         The current agent evaluates his own piece relative to the entire cake.
#         """
# 		return self.piece2d.getRelativeValue()
#
# 	def getEnvy(self, otherPiece):
# 		"""
#         The current agent reports his relative envy of the other agent's piece.
#         """
# 		return self.piece2d.getEnvy(otherPiece)
#
# 	def getLargestEnvy(self, otherPieces):
# 		"""
#         The current agent reports his largest relative envy of another agent's piece.
#         """
# 		return self.piece2d.getLargestEnvy(otherPieces)
#
# 	@lru_cache()
# 	def markQuery(self, value, cutDirection):
# 		return self.piece2d.markQuery(value, cutDirection)
