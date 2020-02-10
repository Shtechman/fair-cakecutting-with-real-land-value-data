import csv
import multiprocessing as mp
import os
import pickle
import re

from utils.Agent import ShadowAgent, Agent
from utils.AllocatedPiece import AllocatedPiece, Piece


class SimulationLog:
    """/**
    * A class that holds a simulation run log data to record a specific simulation.
    */"""

    def __init__(self, result_folder, numberOfAgents, noiseProportion, agent_mapfiles_list, iSimulation,
                 cut_patterns_tested, algName, method, partition, run_duration, comment):

        self.method = method
        self.algName = algName
        self.run_duration = float(run_duration)
        self.comment = comment
        self.process = os.getpid()
        self.result_folder = result_folder
        self.numberOfAgents = numberOfAgents
        self.noiseProportion = noiseProportion
        self.agent_mapfiles_list = agent_mapfiles_list
        self.iSimulation = iSimulation
        self.cuts_tested = [cut_pattern.name for cut_pattern in cut_patterns_tested]

        self.output_csv_file_path = self.result_folder + "logs/" + \
                                    self.iSimulation + "_" + method.split('@')[0] + comment.split('@')[0] + ".csv"
        self.output_log_file_path = self.result_folder + "logs/" + \
                                    self.iSimulation + "_" + method.split('@')[0] + comment.split('@')[0] + ".log"
        self.partition = self._parse_partirion(partition)

        self.printable_partition = [p.toString() for p in partition]
        self.printable_log = self._parse_log()

    def _parse_partirion(self, partition):
        return [{
            'agentName': allocatedPiece.getAgent().getName(),
            'agentMap': allocatedPiece.getAgent().getMapPath(),
            'agentMapNum': allocatedPiece.getAgent().getAgentFileNumber(),
            'iFromRow': allocatedPiece.getIFromRow(),
            'iFromCol': allocatedPiece.getIFromCol(),
            'iToRow': allocatedPiece.getIToRow(),
            'iToCol': allocatedPiece.getIToCol()
        } for allocatedPiece in partition]

    def _parse_log(self):
        return {"Folder": self.result_folder,
                "Number of Agents": self.numberOfAgents,
                "Noise": self.noiseProportion,
                "Cut Patterns Tested": self.cuts_tested,
                "Agent Files": self.agent_mapfiles_list,
                "Experiment": self.iSimulation,
                " ": " ",
                "Method": self.method,
                "Process": self.process,
                "Duration(sec)": self.run_duration,
                "Partition": self.printable_partition}

    def recreate_allocation(self):
        return [AllocatedPiece(Agent(p['agentMap'], p['agentName']),
                               p['iFromRow'],
                               p['iFromCol'],
                               p['iToRow'],
                               p['iToCol']) for p in self.partition]

    def recreate_shadow_allocation(self):
        return [AllocatedPiece(ShadowAgent(p['agentMap'], p['agentName']),
                               p['iFromRow'],
                               p['iFromCol'],
                               p['iToRow'],
                               p['iToCol']) for p in self.partition]

    def recreate_piece_list(self):
        return {p['agentMapNum']: Piece(p['iFromRow'], p['iFromCol'], p['iToRow'], p['iToCol']) for p in self.partition}

    def recreate_shadow_agent_list(self):
        return [ShadowAgent(p['agentMap'], p['agentName']) for p in self.partition]

    def recreate_agent_list(self):
        return [Agent(p['agentMap'], p['agentName']) for p in self.partition]

    def write_to_csv(self):
        with open(self.output_csv_file_path, "w", newline='') as csv_file:
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
            csv_reader = csv.reader(csv_log_file, delimiter=',')
            log_dict = {}
            for row in csv_reader:
                log_dict[row[0]] = row[1]
        folder = log_dict['Folder']
        noise = log_dict['Noise']
        method = log_dict['Method']
        algName = '{}_{}'.format(method.split("_")[0], method.split("_")[1])
        duration = log_dict['Duration(sec)']
        experiment = log_dict['Experiment']
        agent_mapfiles_list = log_dict['Agent Files'].replace('\'', '').replace('[', '').replace(']', '').replace(' ',
                                                                                                                  '').split(
            ',')
        agents = list(map(ShadowAgent, agent_mapfiles_list))
        cuts = log_dict['Partition'].replace('\'', '').replace('receives [', '$').replace('] -', '$').replace('[',
                                                                                                              '').replace(
            ']', '').replace('Anonymous(', '#').replace('Dishonest(', '#').replace(') $', '# $')

        def _parsePartition(p):
            matchObj = re.match(r'#([^#]*)# \$([^\$]*)\$[^\(]* ', p, re.M | re.I)
            return matchObj.group(1), matchObj.group(2)

        cuts_list = [_parsePartition(p) for p in cuts.split('), ')]

        agent_piece_list = []
        for p in cuts_list:
            for agent in agents:
                if p[0] == agent.getAgentFileNumber():
                    agent_piece_list = agent_piece_list + [[agent, p[1]]]

        def _allocatePiece(agent_piece):
            indexes = [float(i) for i in agent_piece[1].split(',')]
            return AllocatedPiece(agent_piece[0], indexes[0], indexes[1], indexes[2], indexes[3])

        partition = list(map(_allocatePiece, agent_piece_list))

        return SimulationLog(folder, len(agents), noise, agent_mapfiles_list, experiment, [], algName, method,
                             partition, duration, "")

    @staticmethod
    def create_logs_from_csv_folder(log_folder_path):
        log_file_list = os.listdir(log_folder_path)
        log_exp_list = list(set([log_name.split('_')[0] for log_name in log_file_list if '.csv' in log_name]))

        print("Sort logs to experiments...")
        log_list_per_exp = {
            exp: [os.path.join(log_folder_path, log_file) for log_file in log_file_list if
                  exp == log_file.split('_')[0]]
            for exp in log_exp_list}
        rlogs = {}
        for exp in log_list_per_exp:
            print("recreating logs for experiment %s" % exp)
            if len(log_list_per_exp[exp]) < 1:
                continue

            p = mp.Pool(4)
            rlogs[exp] = p.map(SimulationLog.create_log_from_csv, log_list_per_exp[exp])
            p.close()
            p.join()

            del p

        return rlogs
