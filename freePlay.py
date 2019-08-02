import csv
import sys

import numpy as np
from utils.SimulationEnvironment import SimulationEnvironment as SimEnv

from utils.Agent import Agent
from utils.MapFileHandler import read_valueMaps_from_csv, plot_partition
from utils.Types import CutPattern, AlgType


def extract_num_of_agents(maps_csv):
    with open(maps_csv, "r", newline='') as csv_file:
        csv_file_reader = csv.reader(csv_file)
        num_of_agents = None
        for line in csv_file_reader:
            if (not line) or line[0].startswith("#"):
                continue
            else:
                num_of_agents = int(line[0])
                break
    return num_of_agents


def extract_cut_patterns(maps_csv):
    with open(maps_csv, "r", newline='') as csv_file:
        csv_file_reader = csv.reader(csv_file)
        cutPattern = []
        for line in csv_file_reader:
            if (not line) or line[0].startswith("#"):
                continue
            elif not line[0].isdigit():
                for entry in line:
                    cutPattern.append(next(value for name,value in vars(CutPattern).items() if name in entry))
                break
            else:
                continue

    return cutPattern


def extract_agents(maps_csv):

    num_of_agents = extract_num_of_agents(testCsvMaps)
    return [Agent(maps_csv, free_play_mode=True, free_play_idx=i) for i in range(num_of_agents)]


def print_results(r, p, agents, plot):
    keys_to_print = ['egalitarianGain', 'utilitarianGain', 'largestEnvy',
                     'smallestFaceRatio', 'averageFaceRatio']
    print("%s Agents in simulation" % r['NumberOfAgents'])
    print("Cut pattern tested %s" % r['Method'])
    print("\n#PARTITION")
    [print(p[i]) for i in range(len(p))]
    print("\n#METRICS")
    [print(key.ljust(25), "| ", r[key]) for key in keys_to_print]

    a = agents[0]
    rows = a.valueMapRows
    cols = a.valueMapCols

    base_map = [[0]*cols]*rows

    """plot a visualization of the simulation"""
    if plot:
        plot_partition(base_map, [str(part) for part in p])


if __name__ == '__main__':

    """ =================== Sim Data Section =================== """

    argv = sys.argv

    """Maps CSV File"""
    if len(argv) > 1:
        testCsvMaps = argv[1]
    else:
        testCsvMaps = 'data/testFolder/freePlay.csv'
    print("Maps taken from %s" % testCsvMaps)
    """-------------------------------------------------"""

    agents = extract_agents(testCsvMaps)

    """Number of Agents"""
    # if not set, num of agents for simulation is as defined in csv file
    if len(argv) > 2:
        num_agents = int(argv[2])
    else:
        num_agents = len(agents)
    """-------------------------------------------------"""

    """Result Folder"""
    if len(argv) > 3:
        testResultPath = argv[3]
    else:
        testResultPath = 'data/testFolder/'
    """-------------------------------------------------"""

    """to stop visualization of results set plot to False"""
    plot = True
    """-------------------------------------------------"""

    """
    Cut patterns:
    Hor,    Ver,    HorVer,    VerHor,    SmallestHalfCut,
    SmallestPiece,    LongestDim,    ShortestDim,    LargestRemainRange,
    MostValuableRemain,    MixedValuableRemain,
    HighestScatter,    SquarePiece
    """

    cut_patterns_to_test = extract_cut_patterns(testCsvMaps)

    """-------------------------------------------------"""
    """-------------------------------------------------"""

    """ ======================================================== """

    """ =================== Sim Execution Section =================== """
    env = SimEnv(0, 0, agents, [], [], testResultPath, cut_patterns_to_test)
    results = []
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    for cut_pattern in cut_patterns_to_test:
        r, p = env.runHonestSimulation(AlgType.EvenPaz, cut_pattern, log=False)
        print_results(r, p, agents, plot)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        results.append(r)
