#!python3
import os
from functools import lru_cache
import numpy as np

from utils.MapFileHandler import read_valueMaps_from_file, read_valueMaps_from_csv
from utils.Types import CutDirection



class Agent:
    """
    an agent has a name and a value-function.
    """
    def __init__(self, valueMapPath, name="Anonymous", free_play_mode=False, free_play_idx=-1):
        self.name = name
        self.hname = name
        self.valueMapPath = valueMapPath
        self.free_play_mode = free_play_mode
        self.file_num = free_play_idx if self.free_play_mode else self.extract_file_name(valueMapPath)
        self.loadValueMap()
        self.cakeValue = np.sum(self.locallyLoadedValueMap)
        self.valueMapRows = len(self.locallyLoadedValueMap)
        self.valueMapCols = len(self.locallyLoadedValueMap[0])
        self.dishonesty = False

    def isDishonest(self):
        return self.dishonesty

    def setDishonesty(self, dishonesty):
        self.dishonesty = dishonesty
        if dishonesty:
            self.name = "Dishonest"
        else:
            self.name = self.hname

    def extract_file_name(self,file_path):
        return file_path.split("/")[-1].split('_')[0]



    def getMapPath(self):
        return self.valueMapPath

    def loadValueMap(self):
        if self.free_play_mode:
            self.locallyLoadedValueMap = np.array(read_valueMaps_from_csv(self.valueMapPath, self.file_num), dtype=np.float)
        else:
            self.locallyLoadedValueMap = np.array(read_valueMaps_from_file(self.valueMapPath), dtype=np.float)

    def cleanMemory(self):
        try:
            del self.locallyLoadedValueMap
        except AttributeError:
            print('passed agent %s clear', self.file_num)

    @lru_cache()
    def valueMapSubsetSum(self, cutsLocations):
        subset = self.valueMapSubset(cutsLocations)
        subsetSum = np.sum(subset)
        del subset
        return subsetSum

    @lru_cache()
    def valueMapSubset(self, cutsLocations):

        iFromRow = cutsLocations[0]
        iFromCol = cutsLocations[1]
        iToRow = cutsLocations[2]
        iToCol = cutsLocations[3]
        if iFromRow < 0 or iFromRow > self.valueMapRows:
            raise ValueError("iFromRow out of range: " + str(iFromRow))
        if iFromCol < 0 or iFromCol > self.valueMapCols:
            raise ValueError("iFromCol out of range: " + str(iFromCol))
        if iToRow < 0 or iToRow > self.valueMapRows:
            raise ValueError("iToRow out of range: " + str(iToRow))
        if iToCol < 0 or iToCol > self.valueMapCols:
            raise ValueError("iToCol out of range: " + str(iToCol))
        if iToRow <= iFromRow or iToCol <= iFromCol:
            return [[]]  # special case not covered by loop below

        fromRowFloor = int(np.floor(iFromRow))
        fromColFloor = int(np.floor(iFromCol))
        toRowCeiling = int(np.ceil(iToRow))
        toColCeiling = int(np.ceil(iToCol))

        fromRowFraction = (fromRowFloor + 1 - iFromRow)
        fromColFraction = (fromColFloor + 1 - iFromCol)
        toRowFraction = 1-(toRowCeiling - iToRow)
        toColFraction = 1-(toColCeiling - iToCol)

        pieceValueMap = self.locallyLoadedValueMap[fromRowFloor:toRowCeiling, fromColFloor:toColCeiling].copy()

        pieceValueMap[0, :] *= fromRowFraction
        pieceValueMap[-1, :] *= toRowFraction
        pieceValueMap[:, 0] *= fromColFraction
        pieceValueMap[:, -1] *= toColFraction
        return pieceValueMap

    @lru_cache()
    def invSumRow(self, iFromRow, iFromCol, iToCol, wanted_sum):
        if iFromRow < 0 or iFromRow > self.valueMapRows:
            raise ValueError("iFromRow out of range: " + str(iFromRow))
        if iFromCol < 0 or iFromCol > self.valueMapCols:
            raise ValueError("iFromCol out of range: " + str(iFromCol))
        if iToCol < 0 or iToCol > self.valueMapCols:
            raise ValueError("iToCol out of range: " + str(iToCol))
        if iToCol <= iFromCol:
            raise ValueError("iToCol out of range: " + str(iToCol))

        fromRowFloor = int(np.floor(iFromRow))
        value = self.valueMapSubsetSum((iFromRow, iFromCol, fromRowFloor+1, iToCol))

        if value >= wanted_sum:
            return iFromRow + (wanted_sum / value)
        wanted_sum -= value
        for i in range(fromRowFloor + 1, self.valueMapRows):
            value = self.valueMapSubsetSum((i, iFromCol, i+1, iToCol))
            if wanted_sum <= value:
                return i + (wanted_sum / value)
            wanted_sum -= value

        return self.valueMapRows

    @lru_cache()
    def invSumCol(self, iFromCol, iFromRow, iToRow, wanted_sum):
        if iFromRow < 0 or iFromRow > self.valueMapRows:
            raise ValueError("iFromRow out of range: " + str(iFromRow))
        if iFromCol < 0 or iFromCol > self.valueMapCols:
            raise ValueError("iFromCol out of range: " + str(iFromCol))
        if iToRow < 0 or iToRow > self.valueMapRows:
            raise ValueError("iToRow out of range: " + str(iToRow))
        if iToRow <= iFromRow:
            raise ValueError("iToRow out of range: " + str(iToRow))

        fromColFloor = int(np.floor(iFromCol))
        value = self.valueMapSubsetSum((iFromRow, iFromCol, iToRow, fromColFloor+1))

        if value >= wanted_sum:
            return iFromCol + (wanted_sum / value)
        wanted_sum -= value
        for i in range(fromColFloor + 1, self.valueMapCols):
            value = self.valueMapSubsetSum((iFromRow, i, iToRow, i+1))
            if wanted_sum <= value:
                return i + (wanted_sum / value)
            wanted_sum -= value

        return self.valueMapCols

    def invSumDirectional(self, iFrom, iRangeFrom, iRangeTo, wanted_sum, cutDirection):
        switcher = {
            CutDirection.Horizontal: self.invSumRow,
            CutDirection.Vertical: self.invSumCol,
        }

        def errorFunc():
            raise ValueError("invalid direction: " + str(cutDirection))

        invSumFunc = switcher.get(cutDirection, errorFunc)
        return invSumFunc(iFrom, iRangeFrom, iRangeTo, wanted_sum)

    @lru_cache()
    def evalQuery(self, cutsLocations):
        # self.loadValueMap()
        vmsSum = self.valueMapSubsetSum(cutsLocations)
        # self.cleanMemory()
        return vmsSum

    @lru_cache()
    def markQuery(self, iFrom, iRangeFrom, iRangeTo, value, direction):
        # self.loadValueMap()
        iTo = self.invSumDirectional(iFrom, iRangeFrom, iRangeTo, value, direction)
        # self.cleanMemory()
        return iTo

    @lru_cache()
    def evaluationOfPiece(self, piece):
        return self.evalQuery(piece.getCuts())

    @lru_cache()
    def evaluationOfCake(self):
        return self.cakeValue




if __name__ == '__main__':

	a = Agent([[1]*8 for _ in range(8)])
	print(a.valueMap)
	print(a.valueMapSubset((0, 0, len(a.valueMap), len(a.valueMap[0]))))
	print(a.valueMapSubset((0, 1, 0, 2)))
	print(a.valueMapSubset((0, 1, 1, 2)))
	print(a.valueMapSubset((0.4, 1, 1, 4)))
	print(a.valueMapSubset((1, 1, 2, 4)))
	print(a.valueMapSubset((1, 1, 1.1, 4)))
	print(a.valueMapSubset((1, 1, 4, 1)))
	print(a.valueMapSubset((1, 1, 4, 2)))
	print(a.valueMapSubset((1, 1, 4, 1.1)))
	print(a.valueMapSubset((1, 1, 6, 4)))
	print(a.valueMapSubset((1,1.5,6,4)))
	print(a.valueMapSubset((0.5,1,6,4)))
	print(a.valueMapSubset((1, 1, 6, 4.5)))
	print(a.valueMapSubset((1, 1, 5.5, 4)))
	print(a.valueMapSubset((1, 1, 5.5, 4.5)))
	print(a.valueMapSubset((1, 1.5, 5.5, 4.5)))
	print(a.valueMapSubset((2.3, 0.6, 3.4, 2.7)))
	print(a.evalQuery((2.3, 0.6, 3.4, 2.7)))
	iTo = a.markQuery(0.2,1,3,6,CutDirection.Horizontal)
	print(iTo)
	vms = a.valueMapSubset((0.2,1,iTo,3))
	print(vms)
	print(np.sum(vms))
	iTo = a.markQuery(0.1,2.5,3.5,5.2,CutDirection.Vertical)
	print(iTo)
	vms = a.valueMapSubset((2.5, 0.1, 3.5, iTo))
	print(vms)
	print(np.sum(vms))
