#!python3

"""
 * @author Erel Segal-Halevi, Gabi Burabia (gabi3b)
 * @since 2016-11
"""

import numpy as np
import matplotlib.pyplot as pyplot
from utils.ValueFunction1D import ValueFunction1D
from utils.AlgorithmEvenPaz1D import AlgorithmEvenPaz1D
from utils.AlgorithmAssessor1D import AlgorithmAssessor1D
from utils.AggregationType import AggregationType
from utils.Agent import Agent
from utils.AllocatedPiece1D import AllocatedPiece1D

np.random.seed(1)

LAND_SIZE = 1000;
VALUE_PER_CELL = 100;
DATA_FILE_NAME ='data/newzealand_forests_npv_4q.1d.json'

meanValues = ValueFunction1D.fromJson(DATA_FILE_NAME);
print("cells in land: %d" % meanValues.length);

algEvenPaz = AlgorithmEvenPaz1D()
algAssessor = AlgorithmAssessor1D(meanValues, algEvenPaz)

def makeSingleExperiment(algorithm, numOfAgents,noiseProportion):
	agents = map(Agent, meanValues.noisyValuesArray(noiseProportion, None, numOfAgents));

	partition = algorithm.run(agents) # returns a list of AllocatedPiece1D
	#print(partition)
	relativeValues = list(map(lambda piece: piece.getRelativeValue(), partition))
	#print(relativeValues)
	egalitarianValue = min(relativeValues)
	egalitarianGain = egalitarianValue*numOfAgents - 1;
	if (egalitarianGain<-0.001): raise ValueError("In proportional division, normalized egalitarian gain must be at least 0; got "+str(egalitarianGain));

	utilitarianValue = sum(relativeValues)
	utilitarianGain = utilitarianValue-1;
	if (utilitarianGain<-0.001): raise ValueError("In proportional division, utilitarian gain must be at least 0; got "+str(utilitarianGain));

	#def getLargestEnvy(piece):
	#	return piece.getLargestEnvy(partition)
	#print(partition[0])
	#print(getLargestEnvy(partition[0]))
	#mostEnviousAgent = max(partition, getLargestEnvy)
	#largestEnvy = mostEnviousAgent.largestEnvy(partition)

	return {
		"numOfAgents": numOfAgents,
		"noiseProportion": noiseProportion,
		"egalitarianGain": egalitarianGain,
		"utilitarianGain": utilitarianGain,
		"largestEnvy": 0,
		}

def calculateSingleDatapoint(numOfAgents,noiseProportion):
	results = []
	for iExperiment in range(EXPERIMENTS_PER_CELL):
		results.append(makeSingleExperiment(algEvenPaz, numOfAgents,noiseProportion))
	return results

def plotResults(results, xAxisName, yAxisName):
	xValues = list(map(lambda result: result[xAxisName], results))
	yValues = list(map(lambda result: result[yAxisName], results))
	pyplot.plot(xValues, yValues)
	pyplot.xlabel(xAxisName)
	pyplot.ylabel(yAxisName)
	pyplot.show()

def calculate_results(aggregationType):
	if aggregationType == AggregationType.NumberOfAgents:
		for numOfAgents in NUMBER_OF_AGENTS:
			print("\n"+str(numOfAgents)+" agents")
			results = []
			for noiseProportion in NOISE_PROPORTION:
				print("\t"+str(noiseProportion)+" noise")
				results += calculateSingleDatapoint(numOfAgents,noiseProportion)
			plotResults(results, xAxisName = "noiseProportion", yAxisName = "egalitarianGain")

	elif aggregationType == AggregationType.Noise:
		for noise in NOISE_PROPORTION:
			print("\n"+str(noise)+" noise")
			results = []
			for numOfAgents in NUMBER_OF_AGENTS:
				print("\t"+str(numOfAgents)+" agents");
				results += calculateSingleDatapoint(numOfAgents,noiseProportion);
			plotResults(results, xAxisName = "noiseProportion", yAxisName = "egalitarianGain")

	else:
		raise Exception("Aggregation Type '%s' is not supported" % aggregationType)


if __name__ == '__main__':
	print("Start experiment")

	EXPERIMENTS_PER_CELL = 1

	NOISE_PROPORTION = [0.2,0.4,0.6,0.8]
	NUMBER_OF_AGENTS = [32]
	calculate_results(AggregationType.NumberOfAgents)

#    NOISE_PROPORTION = [0.2,0.8]
#    NUMBER_OF_AGENTS = [2,4,8,16,32,64,128]
#    calculate_results(AggregationType.Noise)
	print('End experiment')
