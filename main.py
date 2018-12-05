#!python3

"""
 * @author Erel Segal-Halevi, Gabi Burabia (gabi3b),Itay Shtechman
 * @since 2016-11
"""
import itertools
import json
import pickle
import random
import time

import numpy as np

from utils.Agent import Agent
from utils.MapFileHandler import get_valueMaps_from_index
from utils.cakeData2D import CakeData2D
from utils.Plotter import Plotter
from utils.ValueFunction1D import ValueFunction1D
from utils.Types import AggregationType, AlgType, CutDirection, CutPattern
from utils.ExperimentEnvironment import ExperimentEnvironment as ExpEnv
from utils.Measurements import Measurements as Measure


LAND_SIZE = 1000;
VALUE_PER_CELL = 100;
TEST_DATA_FILE_NAME = 'data/testFolder/firstTesttest1_1DVer.txt'
HOR_1D_DATA_FILE_NAME = 'data/‏‏newzealand_forests_1DHor.txt'
VER_1D_DATA_FILE_NAME = 'data/‏‏newzealand_forests_1DVer.txt'
VER_1D_DATA_FILE_NAME_ORIG = 'data/newzealand_forests_npv_4q.1d.json'
TD1_MAP_2D_DATA_FILE_NAME = 'data/testData1_2D.txt'
TD2_MAP_2D_DATA_FILE_NAME = 'data/testData2_2D.txt'

NZ_MAP_2D_DATA_FILE_NAME = 'data/newzealand_forests_2D.txt'
newZealand2D = CakeData2D.fromJson(NZ_MAP_2D_DATA_FILE_NAME)
TestData12D = CakeData2D.fromJson(TD1_MAP_2D_DATA_FILE_NAME)
TestData22D = CakeData2D.fromJson(TD2_MAP_2D_DATA_FILE_NAME)
mapValues = newZealand2D
plotter = Plotter()


def makeSingleExperiment(env, algType, runAssessor, iExperiment):
    print("running for %s agents, algorithm %s, using cut pattern %s" % (env.numberOfAgents, algType.name, env.cutPattern.name))
    if runAssessor:
        partition = env.getAssessor(algType).run(env.getAgents(), env.getCutDirections())  # returns a list of AllocatedPiece
    else:
        partition = env.getAlgorithm(algType).run(env.getAgents(), env.getCutDirections())  # returns a list of AllocatedPiece

    # print(partition)

    # value of piece compared to whole cake (in the eyes of the agent)
    relativeValues = Measure.calculateRelativeValues(partition)

    egalitarianGain = Measure.calculateEgalitarianGain(env.numberOfAgents, relativeValues)

    utilitarianGain = Measure.calculateUtilitarianGain(relativeValues)

    largestEnvy = Measure.calculateLargestEnvy(partition)

    return {
        AggregationType.NumberOfAgents.name: env.numberOfAgents,
        AggregationType.NoiseProportion.name: env.noiseProportion,
        "Algorithm": algType.name,
        "Method": algType.name+env.cutPattern.name,
        "egalitarianGain": egalitarianGain,
        "utilitarianGain": utilitarianGain,
        "largestEnvy": largestEnvy,
        "experiment": iExperiment,
    }


def calculateSingleDatapoint(algType, numOfAgents, noiseProportion):
    results = []

    for iExperiment in range(EXPERIMENTS_PER_CELL):
        print("Fetching %s agents from files" % numOfAgents)
        agents = list(map(Agent, get_valueMaps_from_index("data/newZealandLowResAgents02/index.txt", numOfAgents)))
        # for each experimaent run the Algorithm for numOfAgents using noiseProportion
        env = ExpEnv(noiseProportion, agents, mapValues, CutPattern.Hor)
        results.append(makeSingleExperiment(env, algType, False, iExperiment))
        env = ExpEnv(noiseProportion, agents, mapValues, CutPattern.Ver)
        results.append(makeSingleExperiment(env, algType, False, iExperiment))
        env = ExpEnv(noiseProportion, agents, mapValues, CutPattern.VerHor)
        results.append(makeSingleExperiment(env, algType, False, iExperiment))
        env = ExpEnv(noiseProportion, agents, mapValues, CutPattern.HorVer)
        results.append(makeSingleExperiment(env, algType, False, iExperiment))


    return results


def aggregate(aggregationParams, dataParams, aggText, dataText, dataParamType):
    # create a result graph for each aggregationParam
    for aggParam in aggregationParams:
        print("\n" + aggText + " " + str(aggParam))
        results = calculateMultipleDatapoints(aggParam, AlgType.EvenPaz, dataParamType, dataParams, dataText)
        filename = "./results/" + aggText + "_" + str(aggParam) + ".txt"
        with open(filename, 'w') as json_file:
            json.dump(results, json_file)
        plotter.plotResults(results, list(map(lambda pair: pair[0].name+pair[1].name, list(itertools.product(AlgType, CutPattern)))), xAxisDataType=dataParamType,
                            yAxisData=["largestEnvy"],
                            title="largestEnvy for "+aggText+" "+str(aggParam), experiments=EXPERIMENTS_PER_CELL)
        plotter.plotResults(results, list(
            map(lambda pair: pair[0].name + pair[1].name, list(itertools.product(AlgType, CutPattern)))),
                            xAxisDataType=dataParamType,
                            yAxisData=["utilitarianGain"],
                            title="utilitarianGain for " + aggText + " " + str(aggParam), experiments=EXPERIMENTS_PER_CELL)
        plotter.plotResults(results, list(
            map(lambda pair: pair[0].name + pair[1].name, list(itertools.product(AlgType, CutPattern)))),
                            xAxisDataType=dataParamType,
                            yAxisData=["egalitarianGain"],
                            title="egalitarianGain for " + aggText + " " + str(aggParam), experiments=EXPERIMENTS_PER_CELL)


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
    def createAgents():
        noise = 0.2
        numOfAgents = 256
        original_map = []

        folder = "data/newZealandAgents02"
        input_file = NZ_MAP_2D_DATA_FILE_NAME
        # folder = "data/test02"
        # input_file = TD2_MAP_2D_DATA_FILE_NAME

        indexFile = generate_valueMaps_to_file(input_file, folder, noise, numOfAgents, None)
        # valueMaps = read_valueMaps_from_files(indexFile)
        # generate_agents_to_file(newZealand2D,"newZealand",noise,numOfAgents)

        # rows = len(original_map)
        # cols = len(original_map[0])
        # t1 = time.time()
        # agents_value_functions = [CakeData2D(v,rows,cols) for v in valueMaps]
        # t2 = time.time()
        # print("agents value functions creation was %s seconds" % (t2 - t1))
        # agents = list(map(Agent, agents_value_functions))
        # t3 = time.time()
        # print("agents creation was %s seconds" % (t3 - t2))

    print("Start experiment")
    EXPERIMENTS_PER_CELL = 10

    # NOISE_PROPORTION = [0.2, 0.4, 0.6, 0.8]
    # NUMBER_OF_AGENTS = [8, 16]
    # calculate_results(AggregationType.NumberOfAgents, AlgType.EvanPaz)

    NOISE_PROPORTION = [0.2]
    NUMBER_OF_AGENTS = [4,8,16,32,64,128]
    calculate_results(AggregationType.NoiseProportion)
    print('End experiment')
