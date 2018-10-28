#!python3

def calculateSingleDatapoint(numOfAgents,noiseProportion):
	return [123]

def calculate_results(aggregationType):
	if aggregationType == AggregationType.NumberOfAgents:
		for numOfAgents in NUMBER_OF_AGENTS:
			print("\n"+str(numOfAgents)+" agents")
			results = []
			for noiseProportion in NOISE_PROPORTION:
				print("\t"+str(noiseProportion)+" noise")
				results += calculateSingleDatapoint(numOfAgents,noiseProportion)
			plotResults(results, xAxisName = "noiseProportion")

	elif aggregationType == AggregationType.Noise:
		for noise in NOISE_PROPORTION:
			print("\n"+str(noise)+" noise")
			results = []
			for numOfAgents in NUMBER_OF_AGENTS:
				print("\t"+str(numOfAgents)+" agents");
				results += calculateSingleDatapoint(numOfAgents,noiseProportion)
			plotResults(results, xAxisName = "numOfAgents")

	else:
		raise Exception("Aggregation Type '%s' is not supported" % aggregationType)


if __name__ == '__main__':
	print("Start experiment")

	EXPERIMENTS_PER_CELL = 10

	NOISE_PROPORTION = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
	NUMBER_OF_AGENTS = [32]
	calculate_results(AggregationType.NumberOfAgents)

#    NOISE_PROPORTION = [0.2,0.8]
#    NUMBER_OF_AGENTS = [2,4,8,16,32,64,128]
#    calculate_results(AggregationType.Noise)

	print('End experiment')
