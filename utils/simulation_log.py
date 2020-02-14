import csv
import multiprocessing as mp
import os
import pickle
import re

from utils.agent import ShadowAgent, Agent
from utils.allocated_piece import AllocatedPiece, Piece


class SimulationLog:
    """/**
    * A class that holds a simulation run log data to record a specific simulation.
    *
    * @author Itay Shtechman
    * @since 2019-10
    */"""

    def __init__(
        self,
        result_folder,
        number_of_agents,
        noise_proportion,
        agent_map_files_list,
        i_simulation,
        cut_patterns_tested,
        alg_name,
        method,
        partition,
        run_duration,
        comment,
    ):

        self.method = method
        self.alg_name = alg_name
        self.run_duration = float(run_duration)
        self.comment = comment
        self.process = os.getpid()
        self.result_folder = result_folder
        self.num_of_agents = number_of_agents
        self.noise_proportion = noise_proportion
        self.agent_map_files_list = agent_map_files_list
        self.i_simulation = i_simulation
        self.cuts_tested = [
            cut_pattern.name for cut_pattern in cut_patterns_tested
        ]

        self.output_csv_file_path = (
            self.result_folder
            + "logs/"
            + self.i_simulation
            + "_"
            + method.split("@")[0]
            + comment.split("@")[0]
            + ".csv"
        )
        self.output_log_file_path = (
            self.result_folder
            + "logs/"
            + self.i_simulation
            + "_"
            + method.split("@")[0]
            + comment.split("@")[0]
            + ".log"
        )
        self.partition = self._parse_partirion(partition)

        self.printable_partition = [p.to_string() for p in partition]
        self.printable_log = self._parse_log()

    def _parse_partirion(self, partition):
        return [
            {
                "agentName": allocatedPiece.get_agent().get_name(),
                "agentMap": allocatedPiece.get_agent().get_map_path(),
                "agentMapNum": allocatedPiece.get_agent().get_map_file_number(),
                "i_from_row": allocatedPiece.get_i_from_row(),
                "i_from_col": allocatedPiece.get_i_from_col(),
                "i_to_row": allocatedPiece.get_i_to_row(),
                "i_to_col": allocatedPiece.get_i_to_col(),
            }
            for allocatedPiece in partition
        ]

    def _parse_log(self):
        return {
            "Folder": self.result_folder,
            "Number of Agents": self.num_of_agents,
            "Noise": self.noise_proportion,
            "Cut Patterns Tested": self.cuts_tested,
            "Agent Files": self.agent_map_files_list,
            "Experiment": self.i_simulation,
            " ": " ",
            "Method": self.method,
            "Process": self.process,
            "Duration(sec)": self.run_duration,
            "Partition": self.printable_partition,
        }

    def recreate_allocation(self):
        return [
            AllocatedPiece(
                Agent(p["agentMap"], p["agentName"]),
                p["i_from_row"],
                p["i_from_col"],
                p["i_to_row"],
                p["i_to_col"],
            )
            for p in self.partition
        ]

    def recreate_shadow_allocation(self):
        return [
            AllocatedPiece(
                ShadowAgent(p["agentMap"], p["agentName"]),
                p["i_from_row"],
                p["i_from_col"],
                p["i_to_row"],
                p["i_to_col"],
            )
            for p in self.partition
        ]

    def recreate_piece_list(self):
        return {
            p["agentMapNum"]: Piece(
                p["i_from_row"], p["i_from_col"], p["i_to_row"], p["i_to_col"]
            )
            for p in self.partition
        }

    def recreate_shadow_agent_list(self):
        return [
            ShadowAgent(p["agentMap"], p["agentName"]) for p in self.partition
        ]

    def recreate_agent_list(self):
        return [Agent(p["agentMap"], p["agentName"]) for p in self.partition]

    def write_to_csv(self):
        with open(self.output_csv_file_path, "w", newline="") as csv_file:
            csv_file_writer = csv.writer(csv_file)
            keys_list = self.printable_log.keys()
            data = [[key, self.printable_log[key]] for key in keys_list]
            for data_entry in data:
                csv_file_writer.writerow(data_entry)
        return self.output_csv_file_path

    def write_log_file(self):
        with open(self.output_log_file_path, "wb") as log_file:
            pickle.dump(self, log_file)
        return self.output_log_file_path

    @staticmethod
    def load_log_file(log_file_path):
        with open(log_file_path, "r") as log_file:
            log = pickle.load(log_file)
        return log

    @staticmethod
    def create_log_from_csv(log_file_path):
        with open(log_file_path) as csv_log_file:
            csv_reader = csv.reader(csv_log_file, delimiter=",")
            log_dict = {}
            for row in csv_reader:
                log_dict[row[0]] = row[1]
        folder = log_dict["Folder"]
        noise = log_dict["Noise"]
        method = log_dict["Method"]
        alg_name = "{}_{}".format(method.split("_")[0], method.split("_")[1])
        duration = log_dict["Duration(sec)"]
        experiment = log_dict["Experiment"]
        agent_map_files_list = (
            log_dict["Agent Files"]
            .replace("'", "")
            .replace("[", "")
            .replace("]", "")
            .replace(" ", "")
            .split(",")
        )
        agents = list(map(ShadowAgent, agent_map_files_list))
        cuts = (
            log_dict["Partition"]
            .replace("'", "")
            .replace("receives [", "$")
            .replace("] -", "$")
            .replace("[", "")
            .replace("]", "")
            .replace("Anonymous(", "#")
            .replace("Dishonest(", "#")
            .replace(") $", "# $")
        )

        def _parse_partition(p):
            match_obj = re.match(
                r"#([^#]*)# \$([^\$]*)\$[^\(]* ", p, re.M | re.I
            )
            return match_obj.group(1), match_obj.group(2)

        cuts_list = [_parse_partition(p) for p in cuts.split("), ")]

        agent_piece_list = []
        for p in cuts_list:
            for agent in agents:
                if p[0] == agent.get_map_file_number():
                    agent_piece_list = agent_piece_list + [[agent, p[1]]]

        def _allocate_piece(agent_piece):
            indices = [float(i) for i in agent_piece[1].split(",")]
            return AllocatedPiece(
                agent_piece[0], indices[0], indices[1], indices[2], indices[3]
            )

        partition = list(map(_allocate_piece, agent_piece_list))

        return SimulationLog(
            folder,
            len(agents),
            noise,
            agent_map_files_list,
            experiment,
            [],
            alg_name,
            method,
            partition,
            duration,
            "",
        )

    @staticmethod
    def create_logs_from_csv_folder(log_folder_path):
        log_file_list = os.listdir(log_folder_path)
        log_exp_list = list(
            set(
                [
                    log_name.split("_")[0]
                    for log_name in log_file_list
                    if ".csv" in log_name
                ]
            )
        )

        print("Sort logs to experiments...")
        log_list_per_exp = {
            exp: [
                os.path.join(log_folder_path, log_file)
                for log_file in log_file_list
                if exp == log_file.split("_")[0]
            ]
            for exp in log_exp_list
        }
        rlogs = {}
        for exp in log_list_per_exp:
            print("recreating logs for experiment %s" % exp)
            if len(log_list_per_exp[exp]) < 1:
                continue

            p = mp.Pool(4)
            rlogs[exp] = p.map(
                SimulationLog.create_log_from_csv, log_list_per_exp[exp]
            )
            p.close()
            p.join()

            del p

        return rlogs
