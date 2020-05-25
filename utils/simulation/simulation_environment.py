import csv
import os
from time import time

from utils.algorithms.assessor import AlgorithmSimpleAssessor
from utils.algorithms.strategic import AlgorithmDishonest
from utils.algorithms.even_paz import AlgorithmEvenPaz
from utils.algorithms.focs import AlgorithmFOCS
from utils.algorithms.last_diminisher import AlgorithmLastDiminisher
from utils.maps.map_plotters import plot_all_plots_from_log_object
from utils.simulation.measurements import Measurements as Measure
from utils.simulation.simulation_log import SimulationLog
from utils.ttc.top_trading_cycle import top_trading_cycles
from utils.simulation.cc_types import AlgType, RunType, AggregationType, CutPattern


class SimulationEnvironment:
    """/**
	* A class that holds a simulation environment data required to a specific simulation.
	*
	* @author Itay Shtechman
	* @since 2018-10
	*/"""

    def __init__(
        self,
        i_simulation,
        noise_proportion,
        agents,
        assessor_agent_pool,
        agent_map_files_list,
        result_folder,
        cut_patterns_tested,
    ):
        self.noise_proportion = noise_proportion
        self.agents = agents
        self.num_of_agents = len(agents)
        self.assessor_agent_pool = assessor_agent_pool
        self.agent_map_files_list = agent_map_files_list
        self.result_folder = result_folder
        self.cut_patterns_tested = cut_patterns_tested
        self.i_simulation = i_simulation

    def get_algorithm(self, alg_type, run_type):
        if run_type == RunType.Assessor:
            return self._get_assessor(alg_type)
        if run_type == RunType.Honest:
            return self._get_algorithm(alg_type)
        if run_type == RunType.Dishonest:
            return self._get_dis_algorithm(alg_type)
        else:
            raise ValueError(
                "Algorithm run type '%s' is not supported" % run_type
            )

    def _get_dis_algorithm(self, alg_type):
        return AlgorithmDishonest(self._get_algorithm(alg_type))

    def _get_algorithm(self, alg_type):
        if alg_type == AlgType.EvenPaz:
            return AlgorithmEvenPaz()
        elif alg_type == AlgType.LastDiminisher:
            return AlgorithmLastDiminisher()
        elif alg_type == AlgType.FOCS:
            return AlgorithmFOCS()
        else:
            raise ValueError("Algorithm type '%s' is not supported" % alg_type)

    @staticmethod
    def algorithm_supports_cut_pattern(alg_type, cut_pattern):
        if alg_type == AlgType.EvenPaz:
            return cut_pattern not in [CutPattern.NoPattern]
        elif alg_type == AlgType.LastDiminisher:
            return cut_pattern not in [CutPattern.NoPattern]
        elif alg_type == AlgType.FOCS:
            return cut_pattern in [CutPattern.NoPattern]
        else:
            raise ValueError("Algorithm type '%s' is not supported" % alg_type)

    def _get_assessor(self, alg_type):
        return AlgorithmSimpleAssessor(self.assessor_agent_pool)

    def get_agents(self):
        return self.agents

    def log_simulation_to_file(self, method, partition, run_duration, comment):
        output_file_path = (
            self.result_folder
            + "logs/"
            + self.i_simulation
            + "_"
            + method
            + comment
            + ".csv"
        )

        partition = [p.to_string() for p in partition]
        cuts_tested = [
            cut_pattern.name for cut_pattern in self.cut_patterns_tested
        ]

        log = {
            "Folder": self.result_folder,
            "Number of Agents": self.num_of_agents,
            "Noise": self.noise_proportion,
            "Cut Patterns Tested": cuts_tested,
            "Agent Files": self.agent_map_files_list,
            "Experiment": self.i_simulation,
            " ": " ",
            "Method": method,
            "Process": os.getpid(),
            "Duration(sec)": run_duration,
            "Partition": partition,
        }

        with open(output_file_path, "w", newline="") as csv_file:
            csv_file_writer = csv.writer(csv_file)
            keys_list = log.keys()
            data = [[key, log[key]] for key in keys_list]
            for data_entry in data:
                csv_file_writer.writerow(data_entry)

    def top_trading_cycle_repartition(self, partition, comment="", log=False):
        file_num = lambda a: a.get_map_file_number()
        agents = [p.get_agent() for p in partition]
        initial_ownership = {file_num(p.get_agent()): p for p in partition}
        return self.run_top_trading_cycle(
            agents, initial_ownership, comment, log
        )

    @staticmethod
    def run_top_trading_cycle(
        agents, initial_ownership, comment="", log=False
    ):
        file_num = lambda a: a.get_map_file_number()
        agents_id = {file_num(a) for a in agents}
        pieces_id = {file_num(a) for a in agents}
        agent_preferences = {
            file_num(a): a.piece_by_evaluation(initial_ownership)
            for a in agents
        }
        initial_allocation = {file_num(a): file_num(a) for a in agents}
        new_allocation = top_trading_cycles(
            agents_id, pieces_id, agent_preferences, initial_allocation
        )
        changed = [
            None
            for a in new_allocation
            if not initial_allocation[a] == new_allocation[a]
        ]

        if log and changed:
            print(comment)
            print("New Allocation:", new_allocation)
            print(
                " - old gain",
                {
                    file_num(a): a.evaluation_of_piece(
                        initial_ownership[file_num(a)]
                    )
                    for a in agents
                },
            )
            print(
                " - new gain",
                {
                    file_num(a): a.evaluation_of_piece(
                        initial_ownership[new_allocation[file_num(a)]]
                    )
                    for a in agents
                },
            )
            print()

        return [
            initial_ownership[new_allocation[file_num(a)]].get_allocated_piece(
                a
            )
            for a in agents
        ]

    def parse_results_from_partition(
        self, alg_name, method, partition, run_duration, comment="", log=True
    ):
        if log:
            sim_log = SimulationLog(
                self.result_folder,
                self.num_of_agents,
                self.noise_proportion,
                self.agent_map_files_list,
                self.i_simulation,
                self.cut_patterns_tested,
                alg_name,
                method,
                partition,
                run_duration,
                comment,
            )
            sim_log.write_log_file()
            plot_all_plots_from_log_object(sim_log)

        """ Value of piece compared to whole cake (in the eyes of the agent) """
        partition.sort(key=lambda p: p.get_agent().get_map_file_number())

        relative_values_by_agent = Measure.calculate_relative_values(partition)
        relative_values = relative_values_by_agent.values()
        egalitarian_gain = Measure.calculate_egalitarian_gain(
            self.num_of_agents, relative_values
        )
        utilitarian_gain = Measure.calculate_utilitarian_gain(relative_values)
        largest_envy = Measure.get_largest_envy(partition)

        smallest_face_ratio = Measure.get_smallest_face_ratio(partition)
        average_face_ratio = Measure.get_average_face_ratio(partition)

        ttc_partition = self.top_trading_cycle_repartition(partition)
        ttc_relative_values_by_agent = Measure.calculate_relative_values(
            ttc_partition
        )
        ttc_egalitarian_gain = Measure.calculate_egalitarian_gain(
            self.num_of_agents, ttc_relative_values_by_agent.values()
        )
        ttc_utilitarian_gain = Measure.calculate_utilitarian_gain(
            ttc_relative_values_by_agent.values()
        )
        ttc_largest_envy = Measure.get_largest_envy(ttc_partition)

        return {
            AggregationType.NumberOfAgents.name: self.num_of_agents,
            AggregationType.NoiseProportion.name: self.noise_proportion,
            "Algorithm": alg_name,
            "Method": method
            if "NoPattern" not in method
            else method + comment.split("@")[0],
            "egalitarianGain": egalitarian_gain,
            "ttc_egalitarianGain": ttc_egalitarian_gain,
            "utilitarianGain": utilitarian_gain,
            "ttc_utilitarianGain": ttc_utilitarian_gain,
            "averageFaceRatio": average_face_ratio,
            "largestFaceRatio": 0,        # todo: this field is not used, should be refactored
            "smallestFaceRatio": smallest_face_ratio,
            "averageInheritanceGain": 0,  # todo: this field is not used, should be refactored
            "largestInheritanceGain": 0,  # todo: this field is not used, should be refactored
            "largestEnvy": largest_envy,
            "ttc_largestEnvy": ttc_largest_envy,
            "experimentDurationSec": run_duration,
            "experiment": self.i_simulation,
            "dishonestAgent": comment if "Dishonest" in alg_name else None,
            "relativeValues": relative_values_by_agent,
            "ttc_relativeValues": ttc_relative_values_by_agent,
            "comment": comment
            if "@" not in comment
            else comment.split("@")[1],
        }

    def aggregate_same_simulation_results(self, results):
        result = dict()
        for dkey in results[0].keys():
            if isinstance(results[0][dkey], str):
                result[dkey] = results[0][dkey]
            else:  # number
                if "largestEnvy" in dkey:
                    result[dkey] = min([r[dkey] for r in results])
                else:
                    result[dkey] = max([r[dkey] for r in results])
        return result

    def parse_results_from_partition_list(
        self, alg_name, method, partition, run_duration, log=True
    ):

        if isinstance(partition, dict):
            results = []
            for pkey in partition.keys():
                results.append(
                    self.parse_results_from_partition(
                        alg_name,
                        method,
                        partition[pkey],
                        run_duration,
                        str(pkey),
                        log=log,
                    )
                )
            return results
        else:
            return [
                self.parse_results_from_partition(
                    alg_name, method, partition, run_duration, log=log
                )
            ]

    def run_simulation(self, alg_type, run_type, cut_pattern, log=True):
        time_start = time()
        algorithm = self.get_algorithm(alg_type, run_type)
        partition = algorithm.run(self.get_agents(), cut_pattern)
        time_end = time()

        run_duration = time_end - time_start

        alg_name = algorithm.get_algorithm_type()
        alg_name = (
            "{}_{}".format("Honest", alg_name)
            if run_type is RunType.Honest
            else alg_name
        )

        try:
            method = "{}_{}".format(alg_name, cut_pattern.name)
        except:
            method = "{}_{}".format(alg_name, cut_pattern)

        if isinstance(
            partition, dict
        ):  # multiple partition lists (multiple results)
            run_duration = run_duration / len(partition)

        result = self.parse_results_from_partition_list(
            alg_name, method, partition, run_duration, log=log
        )

        for p in partition:
            del p

        return result

    def run_honest_simulation(self, alg_type, cut_pattern, log=True):
        return self.run_simulation(
            alg_type, RunType.Honest, cut_pattern, log=log
        )

    def run_dishonest_simulation(self, alg_type, cut_pattern, log=True):
        return self.run_simulation(
            alg_type, RunType.Dishonest, cut_pattern, log=log
        )


if __name__ == "__main__":
    print("ok")
