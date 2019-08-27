#!python3

"""
 * @author Erel Segal-Halevi, Gabi Burabia (gabi3b), Itay Shtechman
 * @since 2016-11
"""
import math
import os
import sys
from time import time

from utils.Agent import Agent
from utils.MapFileHandler import get_valueMaps_from_index, get_originalMap_from_index, get_datasetName_from_index
from utils.Plotter import Plotter
from utils.ReportGenerator import create_exp_folder, generate_exp_name, write_results_to_folder, create_run_folder
from utils.Types import AggregationType, AlgType, CutPattern, RunType
from utils.SimulationEnvironment import SimulationEnvironment as SimEnv
import multiprocessing as mp

plotter = Plotter()


def makeSingleSimulation(env, algType, runType, cutPattern):

    print("%s running for %s agents, %s %s algorithm, using cut pattern %s" % (
        os.getpid(), env.numberOfAgents, runType.name, algType.name, cutPattern.name))
    results, partition = env.runSimulation(algType, runType, cutPattern)  # returns a list of AllocatedPiece
    # visualize partition for debugging?
    for p in partition:
        del p
    return results


def runExperiment(exp_data):
    index_file, algType, runTypes, numOfAgents, noiseProportion, iSimulation, assessorAgentPool, result_folder = exp_data
    results = []
    print("======================= %s Agents - PID %s - Simulation %s =======================" % (
    numOfAgents, os.getpid(), iSimulation))
    print("Fetching %s agents from files" % numOfAgents)
    agent_mapfiles_list = get_valueMaps_from_index(index_file, numOfAgents)
    agents = list(map(Agent, agent_mapfiles_list))

    # for each experimaent run the Algorithm for numOfAgents using noiseProportion
    cut_patterns_to_test = [CutPattern.Hor, CutPattern.Ver, CutPattern.HighestScatter, CutPattern.MostValuableRemain,
                            CutPattern.LargestRemainRange, CutPattern.VerHor,
                            CutPattern.HorVer, CutPattern.SmallestPiece, CutPattern.SquarePiece,
                            CutPattern.SmallestHalfCut]

    env = SimEnv(iSimulation, noiseProportion, agents, assessorAgentPool, agent_mapfiles_list, result_folder,
                 cut_patterns_to_test)
    for cur_cut_pattern in cut_patterns_to_test:
        for runType in runTypes:
            for result in makeSingleSimulation(env, algType, runType, cur_cut_pattern):
                results.append(result)
    return results


def calculateSingleDatapoint(index_file, algType, runTypes, numOfAgents, noiseProportion, experiments_per_cell, assessorAgentPool,
                             result_folder):
    p = mp.Pool(NTASKS)

    exp_data = [(index_file, algType, runTypes, numOfAgents, noiseProportion, str(numOfAgents) + str(iSimulation),
                 assessorAgentPool, result_folder) for iSimulation in range(1, experiments_per_cell + 1)]

    result_lists = p.map(runExperiment, exp_data)
    p.close()
    p.join()

    del p

    results = [result for result_list in result_lists for result in result_list]

    return results


def aggregate(index_file, runTypes, aggregationParams, dataParams, aggText, dataText, dataParamType, experiments_per_cell):
    # create a result graph for each aggregationParam
    assessorAgentPool = list(map(Agent, [get_originalMap_from_index(index_file)] * MAX_NUM_OF_AGENTS))
    for aggParam in aggregationParams:
        print("\n" + aggText + " " + str(aggParam))
        exp_name_string = get_datasetName_from_index(index_file) + '_' + generate_exp_name(aggParam, aggText,
                                                                                           experiments_per_cell)
        result_folder = create_exp_folder(RUN_FOLDER_PATH, exp_name_string)

        results = calculateMultipleDatapoints(index_file, aggParam, AlgType.EvenPaz, runTypes, dataParamType, dataParams,
                                              dataText, experiments_per_cell, assessorAgentPool, result_folder)

        write_results_to_folder(result_folder, exp_name_string, results)

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


def calculateMultipleDatapoints(index_file, aggParam, algType, runTypes, dataParamType, dataParams, dataText,
                                experiments_per_cell, assessorAgentPool, result_folder):
    results = []
    # create a data point for each input of dataParam
    for dataParam in dataParams:
        print("\t" + str(dataParam) + " " + dataText)
        if dataParamType == AggregationType.NumberOfAgents:
            results += calculateSingleDatapoint(index_file, algType, runTypes, dataParam, aggParam, experiments_per_cell,
                                                assessorAgentPool, result_folder)
        else:
            results += calculateSingleDatapoint(index_file, algType, runTypes, aggParam, dataParam, experiments_per_cell,
                                                assessorAgentPool, result_folder)
    return results


def calculate_results(index_file, runTypes, aggregationType, number_of_agents, noise_proportion, experiments_per_cell):
    if aggregationType == AggregationType.NumberOfAgents:
        aggregate(index_file, runTypes, number_of_agents, noise_proportion, aggregationType.name, "noise",
                  aggregationType.NoiseProportion, experiments_per_cell)
    elif aggregationType == AggregationType.NoiseProportion:
        aggregate(index_file, runTypes, noise_proportion, number_of_agents, aggregationType.name, "agents",
                  aggregationType.NumberOfAgents, experiments_per_cell)
    else:
        raise Exception("Aggregation Type '%s' is not supported" % aggregationType)


if __name__ == '__main__':
    # main.py [<num_of_experiments> [<num_of_parallel_tasks> [<log_min_num_of_agents> <log_max_num_of_agents>]]]
    print("Start experiment")
    argv = sys.argv
    if len(argv) > 1:
        experiments_per_cell = int(argv[1])
    else:
        experiments_per_cell = 2

    if len(argv) > 2:
        NTASKS = int(argv[2])
    else:
        NTASKS = 4

    if len(argv) > 4:
        log_min_num_of_agents = int(argv[3])
        log_max_num_of_agents = int(argv[4])
        num_of_agents = [int(math.pow(2, y)) for y in range(log_min_num_of_agents, log_max_num_of_agents + 1)]
    else:
        num_of_agents = None

    RUN_FOLDER_PATH = create_run_folder()

    experiment_sets = [
        # {"index_file": "data/newZealandLowResAgents06/index.txt",   "noise_proportion": [0.6],  "num_of_agents": [32, 64, 128], "run_types": [RunType.Honest, RunType.Assessor, RunType.Dishonest]},
        # {"index_file": "data/IsraelMaps06/index.txt",               "noise_proportion": [0.6],  "num_of_agents": [32, 64, 128], "run_types": [RunType.Honest, RunType.Assessor, RunType.Dishonest]},
        # {"index_file": "data/randomMaps06/index.txt",               "noise_proportion": [0.6],  "num_of_agents": [32, 64, 128], "run_types": [RunType.Honest, RunType.Assessor, RunType.Dishonest]},
        # {"index_file": "data/IsraelMaps06/index.txt",             "noise_proportion": [0.6],  "num_of_agents": [64, 128],               "run_types": [RunType.Honest, RunType.Assessor]},
        # {"index_file": "data/IsraelMaps04/index.txt",             "noise_proportion": [0.4],  "num_of_agents": [64, 128],               "run_types": [RunType.Honest, RunType.Assessor]},
        # {"index_file": "data/newZealandLowResAgents06/index.txt", "noise_proportion": [0.6],  "num_of_agents": [64, 128],               "run_types": [RunType.Honest, RunType.Assessor]},
        # {"index_file": "data/newZealandLowResAgents04/index.txt", "noise_proportion": [0.4],  "num_of_agents": [64, 128],               "run_types": [RunType.Honest, RunType.Assessor]},
        # {"index_file": "data/newZealandLowResAgents02/index.txt", "noise_proportion": [0.2],  "num_of_agents": [4, 8, 16, 32, 64, 128], "run_types": [RunType.Honest, RunType.Assessor]},
         {"index_file": "data/IsraelMaps02/index.txt",             "noise_proportion": [0.2],  "num_of_agents": [4, 8], "run_types": [RunType.Honest, RunType.Assessor, RunType.Dishonest]},
        # {"index_file": "data/randomMaps02/index.txt",             "noise_proportion": [0.2],  "num_of_agents": [4, 8, 16, 32, 64, 128], "run_types": [RunType.Honest, RunType.Assessor]},
    ]

    for experiment_set in experiment_sets:
        number_of_agents = num_of_agents if num_of_agents else experiment_set["num_of_agents"]
        MAX_NUM_OF_AGENTS = max(number_of_agents)
        calculate_results(experiment_set["index_file"],
                          experiment_set["run_types"],
                          AggregationType.NoiseProportion,
                          number_of_agents,
                          experiment_set["noise_proportion"],
                          experiments_per_cell)

    print('End experiment')
