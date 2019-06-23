import csv
import os

from utils.AlgorithmAssessor import AlgorithmAssessor
from utils.AlgorithmDishonest import AlgorithmDishonest
from utils.AlgorithmEvenPaz import AlgorithmEvenPaz
from utils.Types import AlgType, CutDirection, RunType
from utils.ValueFunction2D import CakeData2D


class ExperimentEnvironment:
    """/**
	* A class that holds all experiment environment data required to a specific experiment.
	*
	* @author Itay Shtechman
	* @since 2018-10
	*/"""


    def __init__(self, iExperiment, noiseProportion, agents, assessorAgentPool, agent_mapfiles_list, result_folder, cut_patterns_tested):
        self.noiseProportion = noiseProportion
        self.agents = agents
        self.numberOfAgents = len(agents)
        self.assessorAgentPool = assessorAgentPool
        self.agent_mapfiles_list = agent_mapfiles_list
        self.result_folder = result_folder
        self.cut_patterns_tested = cut_patterns_tested
        self.iExperiment = iExperiment


    # def createRandomAgents(self):
    #     agents = map(Agent, self.getMeanValues().noisyValuesArray(self.noiseProportion, None, self.numberOfAgents))
    #     return agents


    def getAlgorithm(self, algType, runType):
        if runType == RunType.Assessor:
            return self._getAssessor(algType)
        if runType == RunType.Honest:
            return self._getAlgorithm(algType)
        if runType == RunType.Dishonest:
            return self._getDisAlgorithm(algType)
        else:
            raise ValueError("Algorithm run type '%s' is not supported" % runType)

    def _getDisAlgorithm(self, algType):
        return AlgorithmDishonest(self._getAlgorithm(algType))

    def _getAlgorithm(self, algType):
        if algType == AlgType.EvenPaz:
            return AlgorithmEvenPaz()
        else:
            raise ValueError("Algorithm type '%s' is not supported" % algType)

    def _getAssessor(self, algType):
        return AlgorithmAssessor(self.assessorAgentPool, self._getAlgorithm(algType))


    def getAgents(self):
        return self.agents

    def log_experiment_to_file(self, method, partition, run_duration,comment):
        output_file_path = self.result_folder + "logs/" + self.iExperiment + "_" + method + comment + ".csv"

        partition = [p.toString() for p in partition]
        cuts_tested = [cut_pattern.name for cut_pattern in self.cut_patterns_tested]

        log = {"Folder": self.result_folder,
               "Number of Agents": self.numberOfAgents,
               "Noise": self.noiseProportion,
               "Cut Patterns Tested": cuts_tested,
               "Agent Files": self.agent_mapfiles_list,
               "Experiment": self.iExperiment,
               " ": " ",
               "Method": method,
               "Process": os.getpid(),
               "Duration(sec)": run_duration,
               "Partition": partition}

        with open(output_file_path, "w", newline='') as csv_file:
            csv_file_writer = csv.writer(csv_file)
            keys_list = log.keys()
            data = [[key, log[key]] for key in keys_list]
            for data_entry in data:
                csv_file_writer.writerow(data_entry)


if __name__ == '__main__':

    # demo test
    env = ExperimentEnvironment(0.02,8,CakeData2D([[1,1,1],[1,1,1],[1,1,1],[1,1,1]],4,3),CutDirection.Vertical)

    print("***")
    [print(agent.valueFunction) for agent in env.getAgents()]
    del env.agents

    print("***")
    [print(agent.valueFunction) for agent in env.getAgents()]
    del env.agents

    print("***")
    [print(agent.valueFunction) for agent in env.getAgents()]
    del env.agents

    print("***")
    [print(agent.valueFunction) for agent in env.getAgents()]
    del env.agents

    print("***")
    [print(agent.valueFunction) for agent in env.getAgents()]

