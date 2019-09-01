"""
Demonstrates that, with n agents, all 2^{n-1} cut-patterns may be needed to get the optimal welfare.

Uses the scenario files in folder exponential-example/*.csv.

Based on: freePlay.py
"""

import csv, re, itertools

from utils.SimulationEnvironment import SimulationEnvironment as SimEnv

from utils.Agent import Agent
from utils.MapFileHandler import read_valueMaps_from_csv, plot_partition
from utils.Types import CutPattern, AlgType, CutDirection


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
        cutPattern_list = False
        freePlay_cuts = False
        freeplay_cuts_list = []
        for line in csv_file_reader:
            if (not line) or line[0].startswith("#"):
                continue
            elif not line[0].isdigit():
                if freePlay_cuts:
                    freeplay_cuts_list.append(line)
                elif "cutPattern" in line:
                    cutPattern_list = True
                elif cutPattern_list:
                    for entry in line:
                        cutPattern.append(next(value for name,value in vars(CutPattern).items() if name in entry))
                    break
                else:
                    freePlay_cuts = True
                    freeplay_cuts_list.append(line)
            elif freePlay_cuts and line[0].isdigit():
                for seq in freeplay_cuts_list:
                    for i,dir in enumerate(seq):
                        if 'hor' in dir.lower():
                            seq[i] = CutDirection.Horizontal
                        elif 'ver' in dir.lower():
                            seq[i] = CutDirection.Vertical
                        else:
                            raise ValueError("did not recognize direction %s, please use either hor or ver." % dir)
                return freeplay_cuts_list
            else:
                continue

    return cutPattern


def extract_agents(maps_csv):
    num_of_agents = extract_num_of_agents(maps_csv)
    return [Agent(maps_csv, free_play_mode=True, free_play_idx=i) for i in range(num_of_agents)]


def print_results(r, p, agents, plot):
    keys_to_print = ['utilitarianGain', 'egalitarianGain', 'largestEnvy',
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
        if isinstance(p, dict):
            for part in p.values():
                plot_partition(base_map, [str(partition) for partition in part])
        else:
            plot_partition(base_map, [str(partition) for partition in p])


h = CutDirection.Horizontal
v = CutDirection.Vertical

if __name__ == '__main__':
    import glob
    dataFolder = "data/exponential-example"
    testCsvMapFiles = glob.glob(dataFolder+"/4-agents-*.csv")
    plot = False  # Set to True to visualize the partitions

    for testCsvMaps in testCsvMapFiles:
        print("\n{}".format(testCsvMaps.replace(dataFolder,"")))
        agents = extract_agents(testCsvMaps)
        num_agents = len(agents)
        cut_patterns_to_test = [list(s) for s in itertools.product([CutDirection.Horizontal,CutDirection.Vertical], repeat=3)]

        testResultPath = 'data/exponential-example/'


        """ =================== Sim Execution Section =================== """
        env = SimEnv(0, 0, agents, [], [], testResultPath, cut_patterns_to_test)
        results = []
        partitions = []
        for cut_pattern in cut_patterns_to_test:
            res, par = env.runHonestSimulation(AlgType.EvenPaz, cut_pattern, log=False)
            for r, p in zip(res,par):
                if plot: print_results(r, par, agents, plot)
                partitions.append(par)
                results.append(r)

        direction_and_gain = []
        for r, p in zip(results, partitions):
            direction_string = '%s,%s,%s' % (tuple(re.findall("Direction.([^:]*)", r['Method'])))
            direction_and_gain.append( (direction_string, r['utilitarianGain']) )
        direction_and_gain.sort(key=lambda pair:-pair[1])
        print("{}".format(direction_and_gain))


