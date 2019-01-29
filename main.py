#!python3

"""
 * @author Erel Segal-Halevi, Gabi Burabia (gabi3b),Itay Shtechman
 * @since 2016-11
"""
import csv
from datetime import datetime
import json
from utils.Agent import Agent
from utils.MapFileHandler import get_valueMaps_from_index, get_originalMap_from_index, read_valueMaps_from_file
from utils.Plotter import Plotter
from utils.Types import AggregationType, AlgType, CutPattern
from utils.ExperimentEnvironment import ExperimentEnvironment as ExpEnv
from utils.Measurements import Measurements as Measure

# TEST_DATA_FILE_NAME = 'data/testFolder/firstTesttest1_1DVer.txt'
# HOR_1D_DATA_FILE_NAME = 'data/‏‏newzealand_forests_1DHor.txt'
# VER_1D_DATA_FILE_NAME = 'data/‏‏newzealand_forests_1DVer.txt'
# VER_1D_DATA_FILE_NAME_ORIG = 'data/newzealand_forests_npv_4q.1d.json'
# TD1_MAP_2D_DATA_FILE_NAME = 'data/testData1_2D.txt'
# TD2_MAP_2D_DATA_FILE_NAME = 'data/testData2_2D.txt'
#
# NZ_MAP_2D_DATA_FILE_NAME = 'data/newzealand_forests_2D.txt'

plotter = Plotter()


def makeSingleExperiment(env, algType, runAssessor, iExperiment, cutPattern):


    if runAssessor:
        algName = "Assessor"
        print("running for %s agents, %s algorithm, using cut pattern %s" % (
        env.numberOfAgents, algName, cutPattern.name))
        partition = env.getAssessor(algType).run(env.getAgents(), cutPattern)  # returns a list of AllocatedPiece
    else:
        algName = algType.name
        print("running for %s agents, %s algorithm, using cut pattern %s" % (
        env.numberOfAgents, algName, cutPattern.name))
        partition = env.getAlgorithm(algType).run(env.getAgents(), cutPattern)  # returns a list of AllocatedPiece

    # print(partition)

    # value of piece compared to whole cake (in the eyes of the agent)
    relativeValues = Measure.calculateRelativeValues(partition)

    egalitarianGain = Measure.calculateEgalitarianGain(env.numberOfAgents, relativeValues)

    utilitarianGain = Measure.calculateUtilitarianGain(relativeValues)

    largestEnvy = Measure.calculateLargestEnvy(partition)

    largestFaceRatio = Measure.calculateLargestFaceRatio(partition)

    averageFaceRatio = Measure.calculateAverageFaceRatio(partition)

    return {
        AggregationType.NumberOfAgents.name: env.numberOfAgents,
        AggregationType.NoiseProportion.name: env.noiseProportion,
        "Algorithm": algName,
        "Method": algName+cutPattern.name,
        "egalitarianGain": egalitarianGain,
        "utilitarianGain": utilitarianGain,
        "averageFaceRatio": averageFaceRatio,
        "largestFaceRatio": largestFaceRatio,
        "largestEnvy": largestEnvy,
        "experiment": iExperiment,
    }


def calculateSingleDatapoint(index_file, algType, numOfAgents, noiseProportion, experiments_per_cell, assessorAgentPool):
    results = []

    for iExperiment in range(1, experiments_per_cell+1):
        print("======================= %s Agents - Experiment %s =======================" % (numOfAgents, iExperiment))
        print("Fetching %s agents from files" % numOfAgents)
        agents = list(map(Agent, get_valueMaps_from_index(index_file, numOfAgents)))

        # for each experimaent run the Algorithm for numOfAgents using noiseProportion
        env = ExpEnv(noiseProportion, agents, assessorAgentPool)
        cut_patterns_to_test = [CutPattern.Hor, CutPattern.Ver, CutPattern.HighestScatter, CutPattern.MostValuableRemain,
                            CutPattern.LargestRemainRange, CutPattern.LargestAvgRemainRange, CutPattern.VerHor,
                            CutPattern.HorVer, CutPattern.SmallestPiece, CutPattern.SquarePiece] # , CutPattern.SmallestHalfCut
        for cut_pattern in cut_patterns_to_test:
            results.append(makeSingleExperiment(env, algType, False, iExperiment, cut_pattern))
            results.append(makeSingleExperiment(env, algType, True, iExperiment, cut_pattern))


    return results


def aggregate(index_file, aggregationParams, dataParams, aggText, dataText, dataParamType, experiments_per_cell):
    # create a result graph for each aggregationParam
    assessorAgentPool = list(map(Agent, [get_originalMap_from_index(index_file)] * MAX_NUM_OF_AGENTS))
    for aggParam in aggregationParams:
        print("\n" + aggText + " " + str(aggParam))
        results = calculateMultipleDatapoints(index_file, aggParam, AlgType.EvenPaz, dataParamType, dataParams, dataText, experiments_per_cell, assessorAgentPool)
        timestring = datetime.now().isoformat(timespec='seconds').replace(":", "-")
        file_name_string = timestring + "_" + aggText + "_" + str(aggParam) + "_" + str(experiments_per_cell) + "_exp"
        jsonfilename = "./results/" + file_name_string + ".json"
        csvfilename = "./results/" + file_name_string + ".csv"

        with open(jsonfilename, "w") as json_file:
            json.dump(results, json_file)

        with open(csvfilename, "w", newline='') as csv_file:
            csv_file_writer = csv.writer(csv_file)
            keys_list = results[0].keys()
            data = [[result[key] for key in keys_list] for result in results]
            csv_file_writer.writerow(keys_list)
            for data_entry in data:
                csv_file_writer.writerow(data_entry)

        """ plotting - don't run plotting if you run it with no human supervision """
        # plotter.plotResults(results, list(map(lambda pair: pair[0].name+pair[1].name, list(itertools.product(AlgType, CutPattern)))), xAxisDataType=dataParamType,
        #                     yAxisData=["largestEnvy"],
        #                     title="largestEnvy for "+aggText+" "+str(aggParam), experiments=experiments_per_cell)
        # plotter.plotResults(results, list(
        #     map(lambda pair: pair[0].name + pair[1].name, list(itertools.product(AlgType, CutPattern)))),
        #                     xAxisDataType=dataParamType,
        #                     yAxisData=["utilitarianGain"],
        #                     title="utilitarianGain for " + aggText + " " + str(aggParam), experiments=experiments_per_cell)
        # plotter.plotResults(results, list(
        #     map(lambda pair: pair[0].name + pair[1].name, list(itertools.product(AlgType, CutPattern)))),
        #                     xAxisDataType=dataParamType,
        #                     yAxisData=["egalitarianGain"],
        #                     title="egalitarianGain for " + aggText + " " + str(aggParam), experiments=experiments_per_cell)


def calculateMultipleDatapoints(index_file, aggParam, algType, dataParamType, dataParams, dataText, experiments_per_cell, assessorAgentPool):
    results = []
    # create a data point for each input of dataParam
    for dataParam in dataParams:
        print("\t" + str(dataParam) + " " + dataText)
        if dataParamType == AggregationType.NumberOfAgents:
            results += calculateSingleDatapoint(index_file, algType, dataParam, aggParam, experiments_per_cell, assessorAgentPool)
        else:
            results += calculateSingleDatapoint(index_file, algType, aggParam, dataParam, experiments_per_cell, assessorAgentPool)
    return results


def calculate_results(index_file, aggregationType, number_of_agents, noise_proportion, experiments_per_cell):
    if aggregationType == AggregationType.NumberOfAgents:
        aggregate(index_file, number_of_agents, noise_proportion, aggregationType.name, "noise", aggregationType.NoiseProportion, experiments_per_cell)
    elif aggregationType == AggregationType.NoiseProportion:
        aggregate(index_file, noise_proportion, number_of_agents, aggregationType.name, "agents", aggregationType.NumberOfAgents, experiments_per_cell)
    else:
        raise Exception("Aggregation Type '%s' is not supported" % aggregationType)

if __name__ == '__main__':

    print("Start experiment")
    number_of_agents = [4, 8, 16, 32, 64, 128]
    MAX_NUM_OF_AGENTS = max(number_of_agents)
    experiments_per_cell = 10

    """ Calculate experiment with Random dataset """
    index_file = "data/randomMaps/index.txt"
    noise_proportion = ['random']
    calculate_results(index_file, AggregationType.NoiseProportion, number_of_agents, noise_proportion,
                      experiments_per_cell)

    """ Calculate experiment with NewZealand 0.2 noise dataset """
    index_file = "data/newZealandLowResAgents02/index.txt"
    noise_proportion = [0.2]
    calculate_results(index_file, AggregationType.NoiseProportion, number_of_agents, noise_proportion,
                      experiments_per_cell)

    """ Calculate experiment with Israel 0.2 noise dataset """
    index_file = "data/IsraelMaps02/index.txt"
    noise_proportion = [0.2]
    calculate_results(index_file, AggregationType.NoiseProportion, number_of_agents, noise_proportion,
                      experiments_per_cell)

    print('End experiment')
