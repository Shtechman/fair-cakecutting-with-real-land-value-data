import json
import sys

from utils.simulation.cc_types import AggregationType, RunType, CutPattern, AlgType
from utils.simulation.simulation_runner import SimulationRunner

if __name__ == "__main__":
    """
    jsonexprunner.py <exp_settings_json_path> [<result_folder_prefix>]

    e.g. > jsonexprunner.py "./batch_files/test.json" "my_test"
    """
    print("Start experiment")
    argv = sys.argv
    if len(argv) > 1:
        json_path = argv[1]
        with open(json_path, "r") as json_file:
            index_file, \
            run_types, \
            agg_type, \
            number_of_agents, \
            noise_proportion, \
            experiments_per_cell, \
            cut_patterns_to_test, \
            alg_types_to_test, \
            run_folder = json.load(json_file)
            agg_type = AggregationType(agg_type)
            run_types = [RunType(rt) for rt in run_types]
            cut_patterns_to_test = [CutPattern(cp) for cp in cut_patterns_to_test]
            alg_types_to_test = [AlgType(at) for at in alg_types_to_test]
    else:
        raise ValueError("No valid json path was provided.")

    if len(argv) > 2:
        folder_name_prefix = argv[2]
    else:
        folder_name_prefix = ""

    sim_runner = SimulationRunner(
        cut_patterns_to_test,
        alg_types_to_test,
        run_folder,
        index_file,
        run_types,
        number_of_agents,
        noise_proportion,
        experiments_per_cell,
        verbosity=True)

    sim_runner.calculate_results(agg_type)

    print("End experiment")
