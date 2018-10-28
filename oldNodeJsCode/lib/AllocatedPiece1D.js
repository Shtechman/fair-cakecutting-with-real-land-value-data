/**
 * A class representing a piece allocated on a 1-dimensional cake.
 *
 * @author Erel Segal-Halevi
 * @since 2014-08
 * @deprecated by AllocatedPiece1D.py
 */

/**
 *  Initialize a 1-dimensional allocated piece based on a value function.
 *  @param valueFunction a ValueFunction1D.
 *  @param iFrom (float) start of allocation.
 *  @param iTo (float) end of allocation.
 */
var AllocatedPiece1D = function(valueFunction,iFrom,iTo) {
	if (isNaN(iFrom)) throw new Error("iFrom is NaN")
	if (isNaN(iTo)) throw new Error("iTo is NaN")
	if (!iFrom) iFrom = 0;
	if (!iTo) iTo = valueFunction.values.length;

	this.valueFunction = valueFunction;
	this.iFrom = iFrom;
	this.iTo = iTo;
}


/*** factory methods ****/

AllocatedPiece1D.fromValueFunction = function(valueFunction) {
	return new AllocatedPiece1D(valueFunction, 0, valueFunction.values.length);
}

AllocatedPiece1D.fromValueFunctionAndBounds = function(valueFunction,iFrom,iTo) {
	return new AllocatedPiece1D(valueFunction, iFrom, iTo);
}



/*** query methods ****/

/**
 * Cut query.
 * @param value what the piece value should be.
 * @return where the piece should end.
 */
AllocatedPiece1D.prototype.getCut = function(value) {
	return this.valueFunction.getCut(this.iFrom, value);
}

AllocatedPiece1D.prototype.getValue = function() {
	return this.valueFunction.getValue(this.iFrom, this.iTo);
}

AllocatedPiece1D.prototype.getValueOfOther = function(other) {
	return this.valueFunction.getValue(other.iFrom, other.iTo);
}

AllocatedPiece1D.prototype.getRelativeValue = function() {
	return this.valueFunction.getRelativeValue(this.iFrom, this.iTo);
}

module.exports = AllocatedPiece1D;
