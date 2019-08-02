import csv
import os
from time import time

from utils.AlgorithmAssessor import AlgorithmAssessor
from utils.AlgorithmDishonest import AlgorithmDishonest
from utils.AlgorithmEvenPaz import AlgorithmEvenPaz
from utils.Types import AlgType, CutDirection, RunType, AggregationType
from utils.ValueFunction2D import CakeData2D
from utils.Measurements import Measurements as Measure


class SimulationEnvironment:
    """/**
	* A class that holds a simulation environment data required to a specific simulation.
	*
	* @author Itay Shtechman
	* @since 2018-10
	*/"""

    def __init__(self, iSimulation, noiseProportion, agents, assessorAgentPool, agent_mapfiles_list, result_folder, cut_patterns_tested):
        self.noiseProportion = noiseProportion
        self.agents = agents
        self.numberOfAgents = len(agents)
        self.assessorAgentPool = assessorAgentPool
        self.agent_mapfiles_list = agent_mapfiles_list
        self.result_folder = result_folder
        self.cut_patterns_tested = cut_patterns_tested
        self.iSimulation = iSimulation


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

    def log_simulation_to_file(self, method, partition, run_duration, comment):
        output_file_path = self.result_folder + "logs/" + self.iSimulation + "_" + method + comment + ".csv"

        partition = [p.toString() for p in partition]
        cuts_tested = [cut_pattern.name for cut_pattern in self.cut_patterns_tested]

        log = {"Folder": self.result_folder,
               "Number of Agents": self.numberOfAgents,
               "Noise": self.noiseProportion,
               "Cut Patterns Tested": cuts_tested,
               "Agent Files": self.agent_mapfiles_list,
               "Experiment": self.iSimulation,
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

    def parseResultsFromPartition(self, algName, method, partition, run_duration, comment="", log=True):
        if log:
            self.log_simulation_to_file(method, partition, run_duration, comment)
        # print(partition)

        # value of piece compared to whole cake (in the eyes of the agent)

        relativeValues = Measure.calculateRelativeValues(partition)
        # relativeValues = Measure.calculateAbsolutValues(partition) # todo - this is just a test, delete this.

        egalitarianGain = Measure.calculateEgalitarianGain(self.numberOfAgents, relativeValues)

        utilitarianGain = Measure.calculateUtilitarianGain(relativeValues)

        largestEnvy = Measure.calculateLargestEnvy(partition)

        largestFaceRatio = Measure.calculateLargestFaceRatio(partition)

        smallestFaceRatio = Measure.calculateSmallestFaceRatio(partition)

        averageFaceRatio = Measure.calculateAverageFaceRatio(partition)

        averageInheritanceGain = Measure.calculateAverageInheritanceGain(self.numberOfAgents, relativeValues)

        largestInheritanceGain = Measure.calculateLargestInheritanceGain(self.numberOfAgents, relativeValues)

        return {
            AggregationType.NumberOfAgents.name: self.numberOfAgents,
            AggregationType.NoiseProportion.name: self.noiseProportion,
            "Algorithm": algName,
            "Method": method,
            "egalitarianGain": egalitarianGain,
            "utilitarianGain": utilitarianGain,
            "averageFaceRatio": averageFaceRatio,
            "largestFaceRatio": largestFaceRatio,
            "smallestFaceRatio": smallestFaceRatio,
            "averageInheritanceGain": averageInheritanceGain,
            "largestInheritanceGain": largestInheritanceGain,
            "largestEnvy": largestEnvy,
            "experimentDurationSec": run_duration,
            "experiment": self.iSimulation,
        }

    def aggregateSameSimulationResults(self, results):
        result = dict()
        for dkey in results[0].keys():
            if isinstance(results[0][dkey], str):
                result[dkey] = results[0][dkey]
            else:  # number
                if 'largestEnvy' in dkey:
                    result[dkey] = min([r[dkey] for r in results])
                else:
                    result[dkey] = max([r[dkey] for r in results])
        return result

    def parseResultsFromPartitionList(self, algName, method, partition, run_duration, log=True):

        if isinstance(partition, dict):
            results = []
            for pkey in partition.keys():
                results.append(self.parseResultsFromPartition(algName, method, partition[pkey], run_duration / self.numberOfAgents,
                                                             "_agent" + pkey, log=log))
            return self.aggregateSameSimulationResults(results)
        else:
            return self.parseResultsFromPartition(algName, method, partition, run_duration, log=log)

    def runSimulation(self, algType, runType, cutPattern, log=True):
        tstart = time()
        partition = self.getAlgorithm(algType, runType).run(self.getAgents(), cutPattern)  # returns a list of AllocatedPiece
        tend = time()

        run_duration = tend - tstart

        algName = "{}_{}".format(runType.name, algType.name)
        method = "{}_{}".format(algName, cutPattern.name)

        result = self.parseResultsFromPartitionList(algName, method, partition, run_duration, log=log)

        return result, partition

    def runHonestSimulation(self, algType, cutPattern, log=True):
        return self.runSimulation(algType, RunType.Honest, cutPattern, log=log)

    def runDishonestSimulation(self, algType, cutPattern, log=True):
        return self.runSimulation(algType, RunType.Dishonest, cutPattern, log=log)


if __name__ == '__main__':
    print('ok')