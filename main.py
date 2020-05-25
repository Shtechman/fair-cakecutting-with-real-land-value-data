#!python3

"""
 * @author Erel Segal-Halevi, Gabi Burabia (gabi3b), Itay Shtechman
 * @since 2016-11
"""
import itertools
import math
import os
import sys

from utils.maps.map_handler import (
    get_value_maps_from_index,
    get_original_map_from_index,
    get_dataset_name_from_index,
)
from utils.reports.report_plotter import Plotter
from utils.reports.report_writer import (
    create_exp_folder,
    generate_exp_name,
    write_results_to_folder,
    create_run_folder,
)
from utils.simulation.cc_types import AggregationType, AlgType, CutPattern, RunType
from utils.simulation.simulation_environment import SimulationEnvironment as SimEnv
from utils.simulation.agent import Agent
import multiprocessing as mp

plotter = Plotter()

""" Static definitions of cut patterns, algorithms and experiment settings to test """
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
# cut_patterns_to_test=[CutPattern.NoPattern,CutPattern.BruteForce,CutPattern.MostValuableMargin,CutPattern.SquarePiece]

alg_types = [AlgType.EvenPaz, AlgType.LastDiminisher, AlgType.FOCS]

experiment_sets = [
    {
        "index_file": "data/IsraelMaps02/index.txt",
        "noise_proportion": [0.2],
        "num_of_agents": [4, 8],
        "run_types": [RunType.Honest],
    },
    # {"index_file": "data/newZealandLowResAgents06/index.txt",
    #  "noise_proportion": [0.6],
    #  "num_of_agents": [4,8,16,32,64,128],
    #  "run_types": [RunType.Honest]},
]
""" -------------------------------------------------------- """


def run_single_simulation(
    env,
    alg_type=AlgType.Simple,
    run_type=RunType.Assessor,
    cut_pattern=CutPattern.NoPattern,
):

    print(
        "%s running for %s agents, %s %s algorithm, using cut pattern %s"
        % (
            os.getpid(),
            env.num_of_agents,
            run_type.name,
            alg_type.name,
            cut_pattern.name,
        )
    )
    results = env.run_simulation(
        alg_type, run_type, cut_pattern
    )  # returns a list of AllocatedPiece
    return results


def run_experiment(exp_data):
    (
        index_file,
        alg_types,
        run_types,
        num_of_agents,
        noise_proportion,
        i_simulation,
        assessor_agent_pool,
        result_folder,
    ) = exp_data
    results = []
    print(
        "======================= %s Agents - PID %s - Simulation %s ======================="
        % (num_of_agents, os.getpid(), i_simulation)
    )
    print("Fetching %s agents from files" % num_of_agents)
    agent_map_files_list = get_value_maps_from_index(index_file, num_of_agents)
    agents = list(map(Agent, agent_map_files_list))

    env = SimEnv(
        i_simulation,
        noise_proportion,
        agents,
        assessor_agent_pool,
        agent_map_files_list,
        result_folder,
        cut_patterns_to_test,
    )
    for cur_cut_pattern in cut_patterns_to_test:
        for algType, runType in itertools.product(alg_types, run_types):
            if env.algorithm_supports_cut_pattern(algType, cur_cut_pattern):
                for result in run_single_simulation(
                    env, algType, runType, cur_cut_pattern
                ):
                    results.append(result)

    assessor_results = run_single_simulation(env)
    for result in assessor_results:
        results.append(result)

    for agent in agents:
        agent.clean_memory()
        del agent

    return results


def calculate_single_datapoint(
    index_file,
    alg_types,
    run_types,
    num_of_agents,
    noise_proportion,
    experiments_per_cell,
    assessor_agent_pool,
    result_folder,
):
    p = mp.Pool(NTASKS)

    exp_data = [
        (
            index_file,
            alg_types,
            run_types,
            num_of_agents,
            noise_proportion,
            str(num_of_agents) + str(i_simulation),
            assessor_agent_pool,
            result_folder,
        )
        for i_simulation in range(1, experiments_per_cell + 1)
    ]

    result_lists = p.map(run_experiment, exp_data)
    p.close()
    p.join()

    del p

    results = [
        result for result_list in result_lists for result in result_list
    ]

    return results


def aggregate(
    index_file,
    run_types,
    aggregation_params,
    data_params,
    agg_text,
    data_text,
    data_param_type,
    experiments_per_cell,
):
    """ Create a result graph for each aggregationParam """
    assessor_agent_pool = list(
        map(
            Agent,
            [get_original_map_from_index(index_file)] * MAX_NUM_OF_AGENTS,
        )
    )

    for agg_param in aggregation_params:
        print("\n" + agg_text + " " + str(agg_param))
        exp_name_string = (
            get_dataset_name_from_index(index_file)
            + "_"
            + generate_exp_name(agg_param, agg_text, experiments_per_cell)
        )
        result_folder = create_exp_folder(RUN_FOLDER_PATH, exp_name_string)

        results = calculate_multiple_datapoints(
            index_file,
            agg_param,
            alg_types,
            run_types,
            data_param_type,
            data_params,
            data_text,
            experiments_per_cell,
            assessor_agent_pool,
            result_folder,
        )

        write_results_to_folder(result_folder, exp_name_string, results)


def calculate_multiple_datapoints(
    index_file,
    agg_param,
    alg_types,
    run_types,
    data_param_type,
    data_params,
    data_text,
    experiments_per_cell,
    assessor_agent_pool,
    result_folder,
):
    results = []
    """ Create a data point for each input of data_param """
    for data_param in data_params:
        print("\t" + str(data_param) + " " + data_text)
        if data_param_type == AggregationType.NumberOfAgents:
            results += calculate_single_datapoint(
                index_file,
                alg_types,
                run_types,
                data_param,
                agg_param,
                experiments_per_cell,
                assessor_agent_pool,
                result_folder,
            )
        else:
            results += calculate_single_datapoint(
                index_file,
                alg_types,
                run_types,
                agg_param,
                data_param,
                experiments_per_cell,
                assessor_agent_pool,
                result_folder,
            )
    return results


def calculate_results(
    index_file,
    run_types,
    aggregation_type,
    num_of_agents,
    noise_proportion,
    experiments_per_cell,
):
    if aggregation_type == AggregationType.NumberOfAgents:
        aggregate(
            index_file,
            run_types,
            num_of_agents,
            noise_proportion,
            aggregation_type.name,
            "noise",
            aggregation_type.NoiseProportion,
            experiments_per_cell,
        )
    elif aggregation_type == AggregationType.NoiseProportion:
        aggregate(
            index_file,
            run_types,
            noise_proportion,
            num_of_agents,
            aggregation_type.name,
            "agents",
            aggregation_type.NumberOfAgents,
            experiments_per_cell,
        )
    else:
        raise Exception(
            "Aggregation Type '%s' is not supported" % aggregation_type
        )


if __name__ == "__main__":
    """
    main.py [<num_of_experiments> [<num_of_parallel_tasks> [<log_min_num_of_agents> <log_max_num_of_agents>]]]
    
    e.g. > main.py 50 4 1 4  -  runs 50 repetitions using 4 threads for agent groups 2,4,8,16
    """
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
        num_of_agents = [
            int(math.pow(2, y))
            for y in range(log_min_num_of_agents, log_max_num_of_agents + 1)
        ]
    else:
        num_of_agents = None

    RUN_FOLDER_PATH = create_run_folder()

    for experiment_set in experiment_sets:
        number_of_agents = (
            num_of_agents if num_of_agents else experiment_set["num_of_agents"]
        )
        MAX_NUM_OF_AGENTS = max(number_of_agents)
        calculate_results(
            experiment_set["index_file"],
            experiment_set["run_types"],
            AggregationType.NoiseProportion,
            number_of_agents,
            experiment_set["noise_proportion"],
            experiments_per_cell,
        )

    print("End experiment")
