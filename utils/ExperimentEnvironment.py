import json
import pickle
from utils.Agent import Agent
from utils.AlgorithmAssessor1D import AlgorithmAssessor1D
from utils.AlgorithmEvenPaz1D import AlgorithmEvenPaz1D
from utils.Types import AlgType, CutDirection, CutPattern
from utils.cakeData2D import CakeData2D


class ExperimentEnvironment:
    """/**
	* A class that holds all experiment environment data required to a specific experiment.
	*
	* @author Itay Shtechman
	* @since 2018-10
	*/"""

    def __init__(self, noiseProportion, agents, mapValues, cutPattern):
        self.noiseProportion = noiseProportion
        self.agents = agents
        self.numberOfAgents = len(agents)
        self.mapValues = mapValues
        self.cutPattern = cutPattern

    def getMeanValues(self):
        return self.mapValues.getAs1D(self.cutDirection)

    # def createRandomAgents(self):
    #     agents = map(Agent, self.getMeanValues().noisyValuesArray(self.noiseProportion, None, self.numberOfAgents))
    #     return agents

    @staticmethod
    def getAlgorithm(algType):
        if algType == AlgType.EvenPaz:
            return AlgorithmEvenPaz1D()
        else:
            raise ValueError("Algorithm type '%s' is not supported" % algType)

    def getAssessor(self, algType):
        return AlgorithmAssessor1D(self.getMeanValues(), self.getAlgorithm(algType))

    def getAgents(self):
        return self.agents

    def getCutDirections(self):
        switcher = {
            CutPattern.Hor: [CutDirection.Horizontal],
            CutPattern.Ver: [CutDirection.Vertical],
            CutPattern.HorVer: [CutDirection.Horizontal, CutDirection.Vertical],
            CutPattern.VerHor: [CutDirection.Vertical, CutDirection.Horizontal],
        }

        return switcher.get(self.cutPattern, None)


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

    # test_path = "./test_agent_as_json.txt"
    # with open(test_path, "wb") as object_file:
    #     a = env.agents
    #     pickle.dump(a, object_file)
    #
    # def _json_angent_hook(d): return Agent(d['valueFunction'], d['name'])
    #
    # with open(test_path, "rb") as object_file:
    #     a = pickle.load(object_file)
    #
    # print(a)
    # print('**')
    # [print(a[i]) for i in range(len(a))]
    # print('**')
    # [print(ag.__dict__) for ag in a]

    #
    # print(a)
    # print('**')
    # [print(a[i]) for i in range(len(a))]
    # print('**')
    # [print(ag.__dict__) for ag in a]
