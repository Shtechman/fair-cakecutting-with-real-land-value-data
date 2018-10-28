#!python3

# This class works only with SageMath

class ValueFunction:
	"""/**
	* A class that represents a 1-dimensional valuation function.
	*
	* @author Erel Segal-Halevi
	* @since 2016-11
	*/"""

	def __init__(self, valueDensityFunction, length):
		"""
		valueDensityFunction is a SageMath function object.
		length is a real number; the cake is [0,length].
		"""
		self.valueDensity = valueDensityFunction
		self.length = length

	def value(self, iFrom, iTo):
		"""
		EXAMPLES::

		sage: f(x) = 1
		sage: Va = ValueFunction(f, 1)
		sage: Va.value(.2, .5)
		0.3

		sage: g(x) = x
		sage: Vb = ValueFunction(g, 1)
		sage: Vb.value(.2, .5)
		0.105

		sage: h(x) = piecewise([ ([0,.3],-3) , ((.3,.4),4) , ([.4,1.0],-6) ])
		sage: Vc = ValueFunction(h, 1)
		sage: Vc.value(.2, .5)
		2.0
		sage: Vc.value(0, 1)
		-5.0
		"""
		return integral(self.valueDensity, (x,iFrom,iTo))

print "ValueFunction class defined"

if __name__ == '__main__':
	import doctest
	doctest.testmod()

	# ValueFunction1D.fromJson("abc.json")
