
from utils.Agent import Agent
from utils.AlgorithmAssessor1D import AlgorithmAssessor1D
from utils.AlgorithmEvenPaz1D import AlgorithmEvenPaz1D
from utils.Types import AlgType


class ExperimentEnvironment:
    """/**
	* A class that holds all experiment environment data required to a specific experiment.
	*
	* @author Itay Shtechman
	* @since 2018-10
	*/"""

    def __init__(self, noiseProportion, numberOfAgents, mapValues, cutDirection):
        self.noiseProportion = noiseProportion
        self.numberOfAgents = numberOfAgents
        self.mapValues = mapValues
        self.cutDirection = cutDirection

    def getMeanValues(self):
        return self.mapValues.getAs1D(self.cutDirection)

    def createRandomAgents(self):
        agents = map(Agent, self.getMeanValues().noisyValuesArray(self.noiseProportion, None, self.numberOfAgents))
        return agents

    @staticmethod
    def getAlgorithm(algType):
        if algType == AlgType.EvenPaz:
            return AlgorithmEvenPaz1D()
        else:
            raise ValueError("Algorithm type '%s' is not supported" % algType)

    def getAssessor(self, algType):
        return AlgorithmAssessor1D(self.getMeanValues(), self.getAlgorithm(algType))

    def getAgents(self):
        try:
            self.agents
        except:
            self.agents = list(self.createRandomAgents())
        return self.agents
