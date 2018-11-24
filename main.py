#!python3

"""
 * @author Erel Segal-Halevi, Gabi Burabia (gabi3b),Itay Shtechman
 * @since 2016-11
"""

import numpy as np

from utils.ValueFunction2D import ValueFunction2D
from utils.Plotter import Plotter
from utils.ValueFunction1D import ValueFunction1D
from utils.Types import AggregationType, AlgType, CutDirection
from utils.ExperimentEnvironment import ExperimentEnvironment as ExpEnv
from utils.Measurements import Measurements as Measure

np.random.seed(1)

LAND_SIZE = 1000;
VALUE_PER_CELL = 100;
TEST_DATA_FILE_NAME = 'data/testFolder/firstTesttest1_1DVer.txt'
HOR_1D_DATA_FILE_NAME = 'data/‏‏newzealand_forests_1DHor.txt'
VER_1D_DATA_FILE_NAME = 'data/‏‏newzealand_forests_1DVer.txt'
VER_1D_DATA_FILE_NAME_ORIG = 'data/newzealand_forests_npv_4q.1d.json'
MAP_2D_DATA_FILE_NAME = 'data/‏‏newzealand_forests_2D.txt'

newZealand2D = ValueFunction2D.fromJson(MAP_2D_DATA_FILE_NAME)
mapValues = newZealand2D
plotter = Plotter()


def makeSingleExperiment(env, algType, runAssessor):

    if runAssessor:
        partition = env.getAssessor(algType).run(env.getAgents())  # returns a list of AllocatedPiece1D
        algName = "Assessor"
    else:
        partition = env.getAlgorithm(algType).run(env.getAgents())  # returns a list of AllocatedPiece1D
        algName = algType.name
    # print(partition)

    # value of piece compared to whole cake (in the eyes of the agent)
    relativeValues = Measure.calculateRelativeValues(partition)

    egalitarianGain = Measure.calculateEgalitarianGain(env.numberOfAgents, relativeValues)

    utilitarianGain = Measure.calculateUtilitarianGain(relativeValues)

    largestEnvy = Measure.calculateLargestEnvy(partition)

    return {
        AggregationType.NumberOfAgents.name: env.numberOfAgents,
        AggregationType.NoiseProportion.name: env.noiseProportion,
        "Algorithm": algName,
        "egalitarianGain": egalitarianGain,
        "utilitarianGain": utilitarianGain,
        "largestEnvy": largestEnvy,
    }


def calculateSingleDatapoint(algType, numOfAgents, noiseProportion):
    results = []

    for iExperiment in range(EXPERIMENTS_PER_CELL):
        # for each experimaent run the Algorithm for numOfAgents using noiseProportion
        env = ExpEnv(noiseProportion, numOfAgents, mapValues, CutDirection.Horizontal)
        results.append(makeSingleExperiment(env, algType, False))
        results.append(makeSingleExperiment(env, algType, True))

    return results


def aggregate(aggregationParams, dataParams, aggText, dataText, dataParamType):
    # create a result graph for each aggregationParam
    for aggParam in aggregationParams:
        print("\n" + aggText + " " + str(aggParam))
        results = calculateMultipleDatapoints(aggParam, AlgType.EvenPaz, dataParamType, dataParams, dataText)
        plotter.plotResults(results, list(map(lambda algType: algType.name, AlgType)), xAxisDataType=dataParamType,
                    yAxisData=["egalitarianGain", "utilitarianGain", "largestEnvy"],
                    title="results for "+aggText+" "+str(aggParam), experiments=EXPERIMENTS_PER_CELL)


def calculateMultipleDatapoints(aggParam, algType, dataParamType, dataParams, dataText):
    results = []
    # create a data point for each input of dataParam
    for dataParam in dataParams:
        print("\t" + str(dataParam) + " " + dataText)
        if dataParamType == AggregationType.NumberOfAgents:
            results += calculateSingleDatapoint(algType, dataParam, aggParam)
        else:
            results += calculateSingleDatapoint(algType, aggParam, dataParam)
    return results


def calculate_results(aggregationType):
    if aggregationType == AggregationType.NumberOfAgents:
        aggregate(NUMBER_OF_AGENTS, NOISE_PROPORTION, aggregationType.name, "noise", aggregationType.NoiseProportion)
    elif aggregationType == AggregationType.NoiseProportion:
        aggregate(NOISE_PROPORTION, NUMBER_OF_AGENTS, aggregationType.name, "agents", aggregationType.NumberOfAgents)
    else:
        raise Exception("Aggregation Type '%s' is not supported" % aggregationType)


if __name__ == '__main__':
    print("Start experiment")

    EXPERIMENTS_PER_CELL = 5

    # NOISE_PROPORTION = [0.2, 0.4, 0.6, 0.8]
    # NUMBER_OF_AGENTS = [8, 16]
    # calculate_results(AggregationType.NumberOfAgents, AlgType.EvanPaz)

    NOISE_PROPORTION = [0.2]
    NUMBER_OF_AGENTS = [2,4,8,16,32]
    calculate_results(AggregationType.NoiseProportion)
    print('End experiment')
