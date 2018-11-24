#!python3

import numpy as np
from functools import lru_cache
import json
import random

from utils.Types import CutDirection
from utils.ValueFunction1D import ValueFunction1D


class ValueFunction2D:
    """/**
	* A class that represents a 2-dimensional piecewise-constant cake.
	*
	* @author Itay Shtechman
	* @since 2018-10
	*/"""

    def __init__(self, values, rows, cols):
        """
		values are the constant values.
		E.g, if values = [[1,3,2],[3,2,1]], then the function equals 1 on [(0,0),(0,1)], 4 on [(0,0),(1,0)] and 8 on [(0,1),(1,2)].
		"""
        self.values = np.array(values)
        self.rows = rows
        self.cols = cols
        self.accuSums = np.array(self.calculateAccuSums())
        self.length = len(values)

    def __repr__(self):
        """
        >>> a = ValueFunction2D([[1,2,3,4],[1,2,3,4]],2,4)
		>>> a
		[[1 2 3 4],[1 2 3 4]]
		>>> str(a)
		'[[1 2 3 4],[1 2 3 4]]'
		"""
        return str(self.values)

    def getHorizontalDim(self):
        return self.rows

    def getVerticalDim(self):
        return self.cols

    def calculateAccuSums(self):
        """
        >>> a = ValueFunction2D([[1,1,1,1],[1,1,1,1],[1,1,1,1]],3,4)
        >>> b = a.calculateAccuSums()
        >>> b
        [[1 2 3 4],[2 4 6 8],[3 6 8 12]]
        """
        accuSums = [[0.0] * self.cols for _ in range(self.rows)]
        for row in range(self.rows):
            if row == 0:
                accuSums[0][0] = self.values[0][0]
            else:
                accuSums[row][0] = self.values[row][0] + accuSums[row - 1][0]
        for col in range(self.cols):
            if col == 0:
                accuSums[0][0] = self.values[0][0]
            else:
                accuSums[0][col] = self.values[0][col] + accuSums[0][col-1]

        for row in range(self.rows):
            if row == 0:
                continue
            for col in range(self.cols):
                if col == 0:
                    continue
                accuSums[row][col] = self.values[row][col] + accuSums[row - 1][col] + accuSums[row][col - 1] - \
                                     accuSums[row - 1][col - 1]

        return accuSums

    @staticmethod
    def fromJson(filename):
        with open(filename) as data_file:
            values = json.load(data_file)
        return ValueFunction2D(values, len(values), len(values[0]))

    def getAs1DHorizontal(self):
        return ValueFunction1D(np.array(self.values).sum(axis=0).tolist())

    def getAs1DVertical(self):
        return ValueFunction1D(np.array(self.values).sum(axis=1).tolist())

    def getAs1D(self, direction):
        switcher = {
            CutDirection.Horizontal: self.getAs1DHorizontal,
            CutDirection.Vertical: self.getAs1DVertical,
        }

        def errorFunc():
            raise ValueError("invalid direction: "+str(direction))

        getter = switcher.get(direction, errorFunc)

        return getter()

    def sumv(self, iFrom, iTo):
        """ /**
		* Given iFrom and iTo, calculate vertical sum of columns in range

		* @param iFrom a float index.
		* @param iTo a float index.
		* @return the sum of the columns between the indices (as float).
		*
		>>> a = ValueFunction2D([[1,1,1,1],[1,1,1,1],[1,1,1,1]],3,4)
		>>> a.sumv(1,3)
		6.0
		>>> a.sumv(1.5,3)
		4.5
		>>> a.sumv(0,2.5)
		7.5
		*
		*/ """

        if iTo <= iFrom:
            return 0.0  # special case not covered by loop below

        return self.sum([0,iFrom,self.rows,iTo])

    def sumh(self, iFrom, iTo):
        """ /**
		* Given iFrom and iTo, calculate horizontal sum of rows in range

		* @param iFrom a float index.
		* @param iTo a float index.
		* @return the sum of the rows between the indices (as float).
		*
		>>> a = ValueFunction2D([[1,1,1,1],[1,1,1,1],[1,1,1,1]],3,4)
		>>> a.sumh(1,1)
		0.0
		>>> a.sumh(1.5,3)
		6.0
		>>> a.sumh(0,2.5)
		10.0
		*
		*/ """

        if iTo <= iFrom:
            return 0.0  # special case not covered by loop below

        return self.sum([iFrom, 0, iTo, self.cols])

    def sumRowRange(self, iRow, iFromCol, iToCol):
        """ /**
        * Given iRow, iFrom and iTo, calculate sum of values in row for a range

        * @param iRow an integer index.
        * @param iFrom a integer index.
        * @param iTo a integer index.
        * @return the sum of the values in row iRow between the indices.
        *
        >>> a = ValueFunction2D([[1,2,3,4],[1,1,1,1],[0,0,0,1]],3,4)
        >>> a.sumRowRange(0,0,1)
        1.0
        >>> a.sumRowRange(1,1,3)
        2.0
        >>> a.sumRowRange(2,0,4)
        1.0
        *
        */ """
        if iFromCol < 0 or iFromCol > self.cols:
            raise ValueError("iFromCol out of range: " + str(iFromCol))
        if iToCol < 0 or iToCol > self.cols:
            raise ValueError("iToCol out of range: " + str(iToCol))
        if iRow < 0 or iRow > self.rows:
            raise ValueError("iRow out of range: " + str(iRow))
        if iToCol < iFromCol:
            return 0.0  # special case not covered by loop below

        sum = 0.0;
        sum += self.accuSums[iRow][iToCol-1]
        if iRow > 0:
            sum -= self.accuSums[iRow-1][iToCol-1]
        if iFromCol > 0:
            sum -= self.accuSums[iRow][iFromCol - 1]
        if iFromCol > 0 and iRow > 0:
            sum += self.accuSums[iRow-1][iFromCol - 1]

        return sum

    def sumColRange(self, iCol, iFromRow, iToRow):
        """ /**
        * Given iCol, iFrom and iTo, calculate sum of values in column for a range

        * @param iCol an integer index.
        * @param iFrom a integer index.
        * @param iTo a integer index.
        * @return the sum of the values in column iCol between the indices.
        *
        >>> a = ValueFunction2D([[1,2,3,4],[1,1,1,1],[0,0,0,1]],3,4)
        >>> a.sumColRange(0,0,3)
        2.0
        >>> a.sumColRange(1,0,2)
        3.0
        >>> a.sumColRange(2,0,3)
        4.0
        *
        */ """
        if iFromRow < 0 or iFromRow > self.rows:
            raise ValueError("iFromRow out of range: " + str(iFromRow))
        if iToRow < 0 or iToRow > self.rows:
            raise ValueError("iToRow out of range: " + str(iToRow))
        if iCol < 0 or iCol > self.cols:
            raise ValueError("iCol out of range: " + str(iCol))
        if iToRow < iFromRow:
            return 0.0  # special case not covered by loop below

        sum = 0.0;
        sum += self.accuSums[iToRow-1][iCol]
        if iCol > 0:
            sum -= self.accuSums[iToRow-1][iCol - 1]
        if iFromRow > 0:
            sum -= self.accuSums[iFromRow - 1][iCol]
        if iCol > 0 and iFromRow > 0:
            sum += self.accuSums[iFromRow - 1][iCol - 1]

        return sum

    def sum(self, cutsLocations):
        """ /**
        * Given cuts locations, calculate sum
        * @param cutsLocations a four float index list.
        * @return the sum of the array between the indices (as float).
        *
        */ """
        iFromRow = cutsLocations[0]
        iFromCol = cutsLocations[1]
        iToRow = cutsLocations[2]
        iToCol = cutsLocations[3]

        if iFromRow < 0 or iFromRow > self.rows:
            raise ValueError("iFromRow out of range: " + str(iFromRow))
        if iFromCol < 0 or iFromCol > self.cols:
            raise ValueError("iFromCol out of range: " + str(iFromCol))
        if iToRow < 0 or iToRow > self.rows:
            raise ValueError("iToRow out of range: " + str(iToRow))
        if iToCol < 0 or iToCol > self.cols:
            raise ValueError("iToCol out of range: " + str(iToCol))
        if iToRow <= iFromRow or iToCol <= iFromCol:
            return 0.0  # special case not covered by loop below

        fromRowFloor = int(np.floor(iFromRow))
        fromRowFraction = (fromRowFloor + 1 - iFromRow)
        fromColFloor = int(np.floor(iFromCol))
        fromColFraction = (fromColFloor + 1 - iFromCol)
        toRowCeiling = int(np.ceil(iToRow))
        toRowCeilingRemovedFraction = (toRowCeiling - iToRow)
        toColCeiling = int(np.ceil(iToCol))
        toColCeilingRemovedFraction = (toColCeiling - iToCol)

        sum = 0.0;
        sum += self.accuSums[toRowCeiling-1][toColCeiling-1]-self.accuSums[toRowCeiling-1][fromColFloor] - \
               self.accuSums[fromRowFloor][toColCeiling-1]+self.accuSums[fromRowFloor][fromColFloor]
        sum += (self.sumRowRange(fromRowFloor, fromColFloor+1, toColCeiling) * fromRowFraction)
        sum += (self.sumColRange(fromColFloor, fromRowFloor+1, toRowCeiling) * fromColFraction)
        sum += (self.values[fromRowFloor][fromColFloor] * fromColFraction * fromRowFraction)
        sum -= (self.sumRowRange(toRowCeiling-1, fromColFloor, toColCeiling-1) * toRowCeilingRemovedFraction)
        sum -= (self.sumColRange(toColCeiling-1, fromRowFloor, toRowCeiling-1) * toColCeilingRemovedFraction)
        sum -= (self.values[toRowCeiling-1][toRowCeiling-1] * toRowCeilingRemovedFraction)
        sum -= (self.values[toRowCeiling-1][toRowCeiling-1] * toColCeilingRemovedFraction)
        sum += (self.values[toRowCeiling-1][toRowCeiling-1] * toColCeilingRemovedFraction * toRowCeilingRemovedFraction)

        return sum

    def invSum(self, iFrom, sum, direction):
        switcher = {
            CutDirection.Horizontal: self.invSumh,
            CutDirection.Vertical: self.invSumv,
        }

        def errorFunc():
            raise ValueError("invalid direction: "+str(direction))

        sumFunc = switcher.get(direction, errorFunc)

        return sumFunc(iFrom, sum)

    def invSumh(self, iFrom, sum):
        return self.invDirectionalSum(iFrom,sum,self.rows,self.sumh)

    def invSumv(self, iFrom, sum):
        return self.invDirectionalSum(iFrom,sum,self.cols,self.sumv)

    def invDirectionalSum(self, iFrom, sum, idxRange, sumFunc):
        """ /**
        * Given iFrom and sum, calculate iTo
        * @param iFrom a float index.
        * @param sum the required sum.
        * @return the final index "iTo", such that sum(values,iFrom,iTo)=sum
        >>> a = ValueFunction1D([1,2,3,4])
        >>> a.invSum(1, 5)
        3.0
        >>> a.invSum(1.5, 4)
        3.0
        >>> a.invSum(1, 6)
        3.25
        >>> a.invSum(1.5,5)
        3.25
        >>>
        *
        */ """
        if iFrom < 0 or iFrom > idxRange:
            raise ValueError("iFrom out of range: " + str(iFrom))
        if sum < 0:
            raise ValueError("sum out of range (should be positive): " + str(sum))

        iFrom = float(iFrom)
        fromFloor = int(np.floor(iFrom));
        fromFraction = (fromFloor + 1 - iFrom);

        value = sumFunc(fromFloor, fromFloor + 1);
        if value * fromFraction >= sum:
            return iFrom + (sum / value);
        sum -= (value * fromFraction);
        for i in range(fromFloor + 1, idxRange):
            value = sumFunc(i, i + 1);
            if sum <= value:
                return i + (sum / value);
            sum -= value;

        # default: returns the largest possible "iTo":
        return idxRange

    @lru_cache()
    def gethCut(self, iFrom, value):
        """/**
         * Cut query.
         * @param from where the piece starts.
         * @param value what the piece value should be.
         * @return where the piece should end.
        */"""
        return self.invSumh(iFrom, value)

    @lru_cache()
    def getvCut(self, iFrom, value):
        """/**
         * Cut query.
         * @param from where the piece starts.
         * @param value what the piece value should be.
         * @return where the piece should end.
        */"""
        return self.invSumv(iFrom, value)

    @lru_cache()
    def valueh(self, iFrom, iTo):
        """/**
         * Eval query
         * @param from where the piece starts.
         * @param to where the piece ends.
         * @return the piece value.
        */"""
        return self.sumh(iFrom, iTo)

    @lru_cache()
    def valuev(self, iFrom, iTo):
        """/**
         * Eval query
         * @param from where the piece starts.
         * @param to where the piece ends.
         * @return the piece value.
        */"""
        return self.sumv(iFrom, iTo)

    @lru_cache()
    def value(self, iFromRow, iFromCol, iToRow, iToCol):
        """/**
         * Eval query
         * @param from where the piece starts.
         * @param to where the piece ends.
         * @return the piece value.
        */"""
        return self.sumv(iFromRow, iFromCol, iToRow, iToCol)

    @lru_cache()
    def getValueOfEntireCake(self):
        """
		>>> a = ValueFunction1D([1,2,3,4])
		>>> a.getValueOfEntireCake()
		10.0
		"""
        return self.sum([0, 0, self.rows, self.cols])

    def getRelativeValueh(self, iFrom, iTo):
        return self.valueh(iFrom, iTo) / self.getValueOfEntireCake()

    def getRelativeValuev(self, iFrom, iTo):
        return self.valuev(iFrom, iTo) / self.getValueOfEntireCake()

    def getRelativeValue(self, iFromRow, iFromCol, iToRow, iToCol):
        return self.value(iFromRow, iFromCol, iToRow, iToCol) / self.getValueOfEntireCake()

    def noisyValues(self, noise_proportion, normalized_sum):
        """/**
		 * @param noise_proportion a number in [0,1]
		 * @return a ValueFunction1D of the same size as self; to each value, the function adds a random noise, drawn uniformly from [-noiseRatio,noiseRatio]*value
		 * @author Erel Segal-Halevi, Gabi Burabia
		 */"""
        aggregated_sum = 0
        values = [[0] * self.cols for _ in range(self.rows)]
        for row in range(self.rows):
            for col in range(self.cols):
                noise = (2 * random.uniform(0,1) - 1) * noise_proportion
                newVal = self.values[row][col] * (1 + noise)
                newVal = max(0, newVal)
                aggregated_sum += newVal
                values[row][col] = newVal
        if aggregated_sum > 0 and normalized_sum is not None and normalized_sum > 0:
            normalization_factor = normalized_sum / aggregated_sum
            values = [values[r][c]*normalization_factor for r in self.rows for c in self.cols]
        return cake(values,self.rows,self.cols)

    def noisyValuesArray(self, noise_proportion, normalized_sum, num_of_agents):
        """
		@return an array of  num_of_agents random ValueFunction1D, uniformly distributed around self.values.
		"""
        valueFunctions = []
        for i in range(num_of_agents):
            valueFunctions.append(self.noisyValues(noise_proportion, normalized_sum))
        return valueFunctions

print("class CakeData2D defined.")  # for debug in sage notebook

if __name__ == '__main__':
    import doctest

    cake = ValueFunction2D([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]], 5, 7)
    print(cake)
    print(cake.accuSums)
    print(cake.sumRowRange(2,0,7))
    print(cake.sumRowRange(2,1,6))
    print(cake.sumColRange(2,1,5))
    print(cake.sumColRange(2,1,2))
    print(cake.sumColRange(2,0,5))
    print("cake.sum(2.5,1,4.5,4)",cake.sum([2.5,1,4.5,4]))
    print(cake.sumh(2,4))
    print(cake.sumv(2,5))
    print("cake.sum(0,0,5,7)",cake.sum([0,0,5,7]))
    print(cake.invSumh(2,cake.sumh(2,4)))
    print(cake.invSumh(0,cake.sumh(0,3)))
    print(cake.invSumh(1,cake.sumh(1,5)))
    print(cake.invSumv(4,cake.sumv(4,6)))
    print(cake.invSumv(0,cake.sumv(0,4)))
    print(cake.invSumv(1,cake.sumv(1,7)))

    a = ValueFunction2D([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], 3, 4)
    print(a.sumv(1, 3))
    print(a.sumv(1.5,3))
    print(a.sumv(0, 2.5))
    print(a.sumh(1, 1))
    print(a.sumh(1.5, 3))
    print(a.sumh(0, 2.5))

    a = ValueFunction2D([[1, 2, 3, 4], [1, 1, 1, 1], [0, 0, 0, 1]], 3, 4)
    print(a.sumRowRange(0, 0, 1))
    print(a.sumRowRange(1, 1, 3))
    print(a.sumRowRange(2, 0, 4))

    print(a.sumColRange(0, 0, 3))
    print(a.sumColRange(1, 0, 2))
    print(a.sumColRange(2, 0, 3))
# ValueFunction1D.fromJson("abc.json")
