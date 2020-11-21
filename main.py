#!python3

"""
 * @author Erel Segal-Halevi, Gabi Burabia (gabi3b), Itay Shtechman
 * @since 2016-11
"""
import math
import sys

from shutil import which

from utils.reports.report_writer import (
    create_run_folder,
)
from utils.simulation.batch_run import schedule_batch_run
from utils.simulation.cc_types import AggregationType, AlgType, CutPattern, RunType
from utils.simulation.simulation_runner import SimulationRunner

""" Static definitions of cut patterns, algorithms and experiment settings to test """
SUPPORT_BATCH = which('sbatch')

cut_patterns_to_test = [
    CutPattern.Hor,
    CutPattern.Ver,
    CutPattern.HighestScatter,
    CutPattern.MostValuableMargin,
    CutPattern.LargestMargin,
    CutPattern.VerHor,
    CutPattern.HorVer,
    CutPattern.SmallestPiece,
    CutPattern.SquarePiece,
    CutPattern.SmallestHalfCut,
    CutPattern.NoPattern,
]
# cut_patterns_to_test = [CutPattern.BruteForce]

alg_types = [AlgType.EvenPaz, AlgType.LastDiminisher, AlgType.FOCS]
# alg_types = [AlgType.EvenPaz, AlgType.LastDiminisher]

experiment_sets = [
    # {"index_file": "data/tlvRealEstate06/index.txt",
    #  "noise_proportion": [0.6],
    #  "num_of_agents": [4, 8],
    #  "run_types": [RunType.Honest]},
    # {"index_file": "data/tlvRealEstate02/index.txt",
    #  "noise_proportion": [0.2],
    #  "num_of_agents": [4, 8],
    #  "run_types": [RunType.Honest]},
    # {"index_file": "data/tlvGardens06/index.txt",
    #  "noise_proportion": [0.6],
    #  "num_of_agents": [4, 8],
    #  "run_types": [RunType.Honest]},
    # {"index_file": "data/IsraelMaps02/index.txt",
    #  "noise_proportion": [0.2],
    #  "num_of_agents": [4, 8],
    #  "run_types": [RunType.Honest]},
    {"index_file": "data/Israel06HS/index.txt",
     "noise_proportion": [0.6],
     "num_of_agents": [4, 8, 16, 32, 64, 128],
     "run_types": [RunType.Honest]},
    {"index_file": "data/newZealandLowRes06HS/index.txt",
     "noise_proportion": [0.6],
     "num_of_agents": [4, 8, 16, 32, 64, 128],
     "run_types": [RunType.Honest]},
    {"index_file": "data/random06HS/index.txt",
     "noise_proportion": [0.6],
     "num_of_agents": [4, 8, 16, 32, 64, 128],
     "run_types": [RunType.Honest]},
]
""" -------------------------------------------------------- """

if __name__ == "__main__":
    """
    main.py [<result_folder_prefix> [<num_of_experiments>
                                        [<num_of_parallel_tasks> [<log_min_num_of_agents> <log_max_num_of_agents>]]]]
    
    e.g. > main.py "" 50 4 1 4  -  runs 50 repetitions using 4 threads for agent groups 2,4,8,16
    """
    print("Start experiment")
    argv = sys.argv
    if len(argv) > 1:
        folder_name_prefix = argv[1]
    else:
        folder_name_prefix = ""

    if len(argv) > 2:
        experiments_per_cell = int(argv[2])
    else:
        experiments_per_cell = 4

    if len(argv) > 3:
        NTASKS = int(argv[3])
    else:
        NTASKS = 4

    if len(argv) > 5:
        log_min_num_of_agents = int(argv[4])
        log_max_num_of_agents = int(argv[5])
        num_of_agents = [
            int(math.pow(2, y))
            for y in range(log_min_num_of_agents, log_max_num_of_agents + 1)
        ]
    else:
        num_of_agents = None

    run_folder_path = create_run_folder(folder_name_prefix)

    for experiment_set in experiment_sets:
        number_of_agents = (
            num_of_agents if num_of_agents else experiment_set["num_of_agents"]
        )
        if SUPPORT_BATCH:
            schedule_batch_run(folder_name_prefix, (
                experiment_set["index_file"],
                [rt.value for rt in experiment_set["run_types"]],
                AggregationType.NoiseProportion.value,
                number_of_agents,
                experiment_set["noise_proportion"],
                experiments_per_cell,
                [cp.value for cp in cut_patterns_to_test],
                [at.value for at in alg_types],
                run_folder_path))
            print("Experiment Scheduled")
        else:
            sim_runner = SimulationRunner(
                cut_patterns_to_test,
                alg_types,
                run_folder_path,
                experiment_set["index_file"],
                experiment_set["run_types"],
                number_of_agents,
                experiment_set["noise_proportion"],
                experiments_per_cell,
                NTASKS,
                True)

            sim_runner.calculate_results(AggregationType.NoiseProportion)
