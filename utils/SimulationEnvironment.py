import csv
import os
from time import time

from utils.AlgorithmAssessor import AlgorithmAssessor
from utils.AlgorithmDishonest import AlgorithmDishonest
from utils.AlgorithmEvenPaz import AlgorithmEvenPaz
from utils.SimulationLog import SimulationLog
from utils.TopTradingCycle import topTradingCycles
from utils.Types import AlgType, RunType, AggregationType
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

    @staticmethod
    def getDishonestAgentFileNum(partition):
        for piece in partition:
            agent = piece.getAgent()
            if agent.isDishonest():
                return agent.file_num
        return None

    def top_trading_cycle_repartition(self, partition, comment="", log=False):
        filenum = lambda a: a.getAgentFileNumber()
        agents = [p.getAgent() for p in partition]
        initialOwnership = {filenum(p.getAgent()): p for p in partition}
        return self.run_top_trading_cycle(agents, initialOwnership, comment, log)

    @staticmethod
    def run_top_trading_cycle(agents, initialOwnership, comment="", log=False):
        filenum = lambda a: a.getAgentFileNumber()
        agents_id = {filenum(a) for a in agents}
        pieces_id = {filenum(a) for a in agents}
        agentPreferences = {filenum(a): a.pieceByEvaluation(initialOwnership) for a in agents}
        initial_allocation = {filenum(a): filenum(a) for a in agents}
        new_allocation = topTradingCycles(agents_id, pieces_id, agentPreferences, initial_allocation)
        changed = [None for a in new_allocation if not initial_allocation[a] == new_allocation[a]]

        if log and changed:
            print(comment)
            print('New Allocation:', new_allocation)
            print(' - old gain', {filenum(a): a.evaluationOfPiece(initialOwnership[filenum(a)]) for a in agents})
            print(' - new gain',
                  {filenum(a): a.evaluationOfPiece(initialOwnership[new_allocation[filenum(a)]]) for a in agents})
            print()


        return [initialOwnership[new_allocation[filenum(a)]].getAllocatedPiece(a) for a in agents]

    def parseResultsFromPartition(self, algName, method, partition, run_duration, comment="", log=True):
        if log:
            simLog = SimulationLog(self.result_folder,self.numberOfAgents,self.noiseProportion,self.agent_mapfiles_list,
                                   self.iSimulation,self.cut_patterns_tested, algName, method, partition, run_duration, comment)
            simLog.write_log_file()
        # print(partition)

        # value of piece compared to whole cake (in the eyes of the agent)

        relativeValuesByAgent = Measure.calculateRelativeValues(partition)
        relativeValues = relativeValuesByAgent.values()
        egalitarianGain = Measure.calculateEgalitarianGain(self.numberOfAgents, relativeValues)
        utilitarianGain = Measure.calculateUtilitarianGain(relativeValues)
        largestEnvy = Measure.calculateLargestEnvy(partition)

        largestFaceRatio = Measure.calculateLargestFaceRatio(partition)

        smallestFaceRatio = Measure.calculateSmallestFaceRatio(partition)

        averageFaceRatio = Measure.calculateAverageFaceRatio(partition)

        averageInheritanceGain = Measure.calculateAverageInheritanceGain(self.numberOfAgents, relativeValues)

        largestInheritanceGain = Measure.calculateLargestInheritanceGain(self.numberOfAgents, relativeValues)

        dishonestAgent = self.getDishonestAgentFileNum(partition)

        ttc_partition = self.top_trading_cycle_repartition(partition)
        ttc_relativeValuesByAgent = Measure.calculateRelativeValues(ttc_partition)
        ttc_egalitarianGain = Measure.calculateEgalitarianGain(self.numberOfAgents, ttc_relativeValuesByAgent.values())
        ttc_utilitarianGain = Measure.calculateUtilitarianGain(ttc_relativeValuesByAgent.values())
        ttc_largestEnvy = Measure.calculateLargestEnvy(ttc_partition)

        return {
            AggregationType.NumberOfAgents.name: self.numberOfAgents,
            AggregationType.NoiseProportion.name: self.noiseProportion,
            "Algorithm": algName,
            "Method": method,
            "egalitarianGain": egalitarianGain,
            "ttc_egalitarianGain": ttc_egalitarianGain,
            "utilitarianGain": utilitarianGain,
            "ttc_utilitarianGain": ttc_utilitarianGain,
            "averageFaceRatio": averageFaceRatio,
            "largestFaceRatio": largestFaceRatio,
            "smallestFaceRatio": smallestFaceRatio,
            "averageInheritanceGain": averageInheritanceGain,
            "largestInheritanceGain": largestInheritanceGain,
            "largestEnvy": largestEnvy,
            "ttc_largestEnvy": ttc_largestEnvy,
            "experimentDurationSec": run_duration,
            "experiment": self.iSimulation,
            "dishonestAgent": dishonestAgent,
            "relativeValues": relativeValuesByAgent,
            "ttc_relativeValues": ttc_relativeValuesByAgent,
            "comment": comment,
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
                results.append(self.parseResultsFromPartition(algName, method, partition[pkey], run_duration,
                                                              str(pkey), log=log))
            return results
        else:
            return [self.parseResultsFromPartition(algName, method, partition, run_duration, log=log)]

    def runSimulation(self, algType, runType, cutPattern, log=False):
        tstart = time()
        algorithm = self.getAlgorithm(algType, runType)
        partition = algorithm.run(self.getAgents(), cutPattern)  # returns a list of AllocatedPiece
        tend = time()

        run_duration = tend - tstart

        algName = algorithm.getAlgorithmType()
        algName = "{}_{}".format("Honest", algName) if runType is RunType.Honest else algName

        try:
            method = "{}_{}".format(algName, cutPattern.name)
        except:
            method = "{}_{}".format(algName, cutPattern)

        if isinstance(partition, dict):  # multiple partition lists (multiple results)
            run_duration = run_duration/len(partition)

        result = self.parseResultsFromPartitionList(algName, method, partition, run_duration, log=log)

        for p in partition:
            del p

        return result

    def runHonestSimulation(self, algType, cutPattern, log=True):
        return self.runSimulation(algType, RunType.Honest, cutPattern, log=log)

    def runDishonestSimulation(self, algType, cutPattern, log=True):
        return self.runSimulation(algType, RunType.Dishonest, cutPattern, log=log)


if __name__ == '__main__':
    print('ok')