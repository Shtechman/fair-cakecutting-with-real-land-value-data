#!python3

"""
 * @author Itay Shtechman
 * @since 2020-06
"""

import itertools
import os

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
)
from utils.simulation.cc_types import AggregationType, AlgType, CutPattern, RunType
from utils.simulation.simulation_environment import SimulationEnvironment as SimEnv
from utils.simulation.agent import Agent
import multiprocessing as mp

plotter = Plotter()


class SimulationRunner:

    def __init__(self,
                 cut_patterns_to_test,
                 alg_types_to_test,
                 run_folder,
                 index_file,
                 run_types,
                 num_of_agents,
                 noise_proportion,
                 experiments_per_cell,
                 NTASKS=4,
                 verbosity=False):
        self.cut_patterns_to_test = cut_patterns_to_test
        self.alg_types_to_test = alg_types_to_test
        self.run_folder = run_folder
        self.index_file = index_file
        self.run_types = run_types
        self.num_of_agents = num_of_agents
        self.noise_proportion = noise_proportion
        self.experiments_per_cell = experiments_per_cell
        self.NTASKS = NTASKS
        self.verbosity = verbosity

    def run_single_simulation(
            self,
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
            alg_type, run_type, cut_pattern, log=self.verbosity
        )
        return results

    def run_experiment(self, exp_data):
        (
            _index_file,
            _alg_types,
            _run_types,
            _num_of_agents,
            _noise_proportion,
            _i_simulation,
            _assessor_agent_pool,
            _result_folder,
        ) = exp_data
        results = []
        print(
            "======================= %s Agents - PID %s - Simulation %s ======================="
            % (_num_of_agents, os.getpid(), _i_simulation)
        )
        print("Fetching %s agents from files" % _num_of_agents)
        agent_map_files_list = get_value_maps_from_index(_index_file, _num_of_agents)
        agents = list(map(Agent, agent_map_files_list))

        env = SimEnv(
            _i_simulation,
            _noise_proportion,
            agents,
            _assessor_agent_pool,
            agent_map_files_list,
            _result_folder,
            self.cut_patterns_to_test,
        )
        for cur_cut_pattern in self.cut_patterns_to_test:
            for algType, runType in itertools.product(_alg_types, _run_types):
                if env.algorithm_supports_cut_pattern(algType, cur_cut_pattern):
                    for result in self.run_single_simulation(
                            env, algType, runType, cur_cut_pattern
                    ):
                        results.append(result)

        assessor_results = self.run_single_simulation(env)
        highest_bidder_results = self.run_single_simulation(env, AlgType.HighestBidder, RunType.Honest)
        for result in assessor_results:
            results.append(result)

        results.append(highest_bidder_results)

        for agent in agents:
            agent.clean_memory()
            del agent

        return results

    def calculate_single_datapoint(
            self,
            num_of_agents,
            noise_proportion,
            assessor_agent_pool
    ):
        p = mp.Pool(self.NTASKS)

        exp_data = [
            (
                self.index_file,
                self.alg_types_to_test,
                self.run_types,
                num_of_agents,
                noise_proportion,
                str(num_of_agents) + str(i_simulation),
                assessor_agent_pool,
                self.result_folder,
            )
            for i_simulation in range(1, self.experiments_per_cell + 1)
        ]

        result_lists = p.map(self.run_experiment, exp_data)
        p.close()
        p.join()

        del p

        results = [
            result for result_list in result_lists for result in result_list
        ]

        return results

    def aggregate(
            self,
            aggregation_params,
            data_params,
            agg_text,
            data_text,
            data_param_type
    ):
        """ Create a result graph for each aggregationParam """
        assessor_agent_pool = list(
            map(
                Agent,
                [get_original_map_from_index(self.index_file)] * max(self.num_of_agents),
            )
        )

        for agg_param in aggregation_params:
            print("\n" + agg_text + " " + str(agg_param))
            exp_name_string = (
                    get_dataset_name_from_index(self.index_file)
                    + "_"
                    + generate_exp_name(agg_param, agg_text, self.experiments_per_cell)
            )
            self.result_folder = create_exp_folder(self.run_folder, exp_name_string)

            results = self.calculate_multiple_datapoints(
                agg_param,
                data_param_type,
                data_params,
                data_text,
                assessor_agent_pool
            )

            write_results_to_folder(self.result_folder, exp_name_string, results)

    def calculate_multiple_datapoints(
            self,
            agg_param,
            data_param_type,
            data_params,
            data_text,
            assessor_agent_pool
    ):
        results = []
        """ Create a data point for each input of data_param """
        for data_param in data_params:
            print("\t" + str(data_param) + " " + data_text)
            if data_param_type == AggregationType.NumberOfAgents:
                results += self.calculate_single_datapoint(
                    data_param,
                    agg_param,
                    assessor_agent_pool
                )
            else:
                results += self.calculate_single_datapoint(
                    agg_param,
                    data_param,
                    assessor_agent_pool
                )
        return results

    def calculate_results(self, aggregation_type):
        if aggregation_type == AggregationType.NumberOfAgents:
            self.aggregate(
                self.num_of_agents,
                self.noise_proportion,
                aggregation_type.name,
                "noise",
                aggregation_type.NoiseProportion)
        elif aggregation_type == AggregationType.NoiseProportion:
            self.aggregate(
                self.noise_proportion,
                self.num_of_agents,
                aggregation_type.name,
                "agents",
                aggregation_type.NumberOfAgents)
        else:
            raise Exception(
                "Aggregation Type '%s' is not supported" % aggregation_type
            )
