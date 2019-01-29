
from utils.AlgorithmAssessor1D import AlgorithmAssessor1D
from utils.AlgorithmEvenPaz1D import AlgorithmEvenPaz1D
from utils.Types import AlgType, CutDirection
from utils.ValueFunction2D import CakeData2D


class ExperimentEnvironment:
    """/**
	* A class that holds all experiment environment data required to a specific experiment.
	*
	* @author Itay Shtechman
	* @since 2018-10
	*/"""

    def __init__(self, noiseProportion, agents, assessorAgentPool):
        self.noiseProportion = noiseProportion
        self.agents = agents
        self.numberOfAgents = len(agents)
        self.assessorAgentPool = assessorAgentPool


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
        return AlgorithmAssessor1D(self.assessorAgentPool, self.getAlgorithm(algType))

    def getAgents(self):
        return self.agents


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

