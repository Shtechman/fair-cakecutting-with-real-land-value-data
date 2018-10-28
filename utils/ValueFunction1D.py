#!python3

import numpy as np
from functools import lru_cache
import json

class ValueFunction1D:
	"""/**
	* A class that represents a 1-dimensional piecewise-constant function.
	*
	* @author Erel Segal-Halevi
	* @since 2016-11
	*/"""

	def __init__(self, values):
		"""
		values are the constant values.
		E.g, if values = [1,3,2], then the function equals 1 on [0,1], 3 on [1,2] and 2 on [2,3].
		"""
		self.values = np.array(values)
		self.length = len(values)

	def __repr__(self):
		"""
		>>> a = ValueFunction1D([1,2,3,4])
		>>> a
		[1 2 3 4]
		>>> str(a)
		'[1 2 3 4]'
		"""
		return str(self.values)

	@staticmethod
	def fromJson(filename):
		with open(filename) as data_file:
			values = json.load(data_file)
		return ValueFunction1D(values)

	def sum(self, iFrom, iTo):
		""" /**
		* Given iFrom and iTo, calculate sum
		* @param iFrom a float index.
		* @param iTo a float index.
		* @return the sum of the array between the indices (as float).
		*
		>>> a = ValueFunction1D([1,2,3,4])
		>>> a.sum(1,3)
		5.0
		>>> a.sum(1.5,3)
		4.0
		>>> a.sum(1,3.25)
		6.0
		>>> a.sum(1.5,3.25)
		5.0
		>>> a.sum(3,3)
		0.0
		>>>
		*
		*/ """
		if iFrom<0 or iFrom>self.length:
			raise ValueError("iFrom out of range: "+str(iFrom))
		if iTo<0 or iTo>self.length:
			raise ValueError("iTo out of range: "+str(iTo))
		if iTo<=iFrom:
			return 0.0  # special case not covered by loop below

		fromFloor = int(np.floor(iFrom))
		fromFraction = (fromFloor+1-iFrom)
		toCeiling = int(np.ceil(iTo))
		toCeilingRemovedFraction = (toCeiling-iTo)

		sum = 0.0;
		sum += (self.values[fromFloor]*fromFraction)
		sum += self.values[fromFloor+1:toCeiling].sum()
		sum -= (self.values[toCeiling-1]*toCeilingRemovedFraction)

		return sum

	def invSum(self, iFrom, sum):
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
		if iFrom<0 or iFrom>self.length:
			raise ValueError("iFrom out of range: "+str(iFrom))
		if sum<0:
			raise ValueError("sum out of range (should be positive): "+str(sum))

		iFrom = float(iFrom)
		fromFloor = int(np.floor(iFrom));
		fromFraction = (fromFloor+1-iFrom);

		value = self.values[fromFloor];
		if value*fromFraction >= sum:
			return iFrom + (sum/value);
		sum -= (value*fromFraction);
		for i in range(fromFloor+1, self.length):
			value = self.values[i];
			if sum <= value:
				return i + (sum/value);
			sum -= value;

		# default: returns the largest possible "iTo":
		return values.length

	@lru_cache()
	def getCut(self, iFrom, value):
		"""/**
		 * Cut query.
		 * @param from where the piece starts.
		 * @param value what the piece value should be.
		 * @return where the piece should end.
		*/"""
		return self.invSum(iFrom, value)

	@lru_cache()
	def value(self, iFrom, iTo):
		"""/**
		 * Eval query
		 * @param from where the piece starts.
		 * @param to where the piece ends.
		 * @return the piece value.
		*/"""
		return self.sum(iFrom, iTo)

	@lru_cache()
	def getValueOfEntireCake(self):
		"""
		>>> a = ValueFunction1D([1,2,3,4])
		>>> a.getValueOfEntireCake()
		10.0
		"""
		return self.sum(0, self.length)

	def getRelativeValue(self, iFrom, iTo):
		return self.value(iFrom,iTo) / self.getValueOfEntireCake()

	def noisyValues(self, noise_proportion, normalized_sum):
		"""/**
		 * @param noise_proportion a number in [0,1]
		 * @return a ValueFunction1D of the same size as self; to each value, the function adds a random noise, drawn uniformly from [-noiseRatio,noiseRatio]*value
		 * @author Erel Segal-Halevi, Gabi Burabia
		 */"""
		aggregated_sum = 0
		values = [0] * self.length
		for i in range(self.length):
			noise = (2*np.random.rand()-1)*noise_proportion
			newVal = self.values[i]*(1+noise)
			newVal = max(0, newVal)
			aggregated_sum += newVal
			values[i] = newVal
		if aggregated_sum > 0 and normalized_sum is not None and normalized_sum > 0:
			normalization_factor = normalized_sum / aggregated_sum
			for i in range(len(values)):
				values[i] *= normalization_factor
		return ValueFunction1D(values)

	def noisyValuesArray(self, noise_proportion, normalized_sum, num_of_agents):
		"""
		@return an array of  num_of_agents random ValueFunction1D, uniformly distributed around self.values.
		"""
		valueFunctions = []
		for i in range(num_of_agents):
			valueFunctions.append(self.noisyValues(noise_proportion, normalized_sum))
		return valueFunctions

	def partitionValues(self, cutPoints):
		"""
		>>> a = ValueFunction1D([1,2,3,4])
		>>> a.partitionValues([1,2])
		[1.0, 2.0, 7.0]
		>>> a.partitionValues([3,3])
		[6.0, 0.0, 4.0]
		"""
		values = []
		values.append(self.value(0, cutPoints[0]))
		for i in range(len(cutPoints)-1):
			values.append(self.value(cutPoints[i], cutPoints[i+1]))
		values.append(self.value(cutPoints[-1],self.length))
		return values

	def partitionBestPiece(self, cutPoints):
		"""
		>>> a = ValueFunction1D([1,2,3,4])
		>>> a.partitionBestPiece([1,2])
		2
		>>> a.partitionBestPiece([3,3])
		0
		"""
		values = self.partitionValues(cutPoints)
		return np.argmax(values)

print("class ValueFunction1D defined.") # for debug in sage notebook

if __name__ == '__main__':
	import doctest
	doctest.testmod()

	# ValueFunction1D.fromJson("abc.json")
