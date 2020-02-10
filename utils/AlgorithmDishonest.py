#!python3

"""
/**
 * Implementation of the dishonest proportional cake-cutting algorithm.
 *
 */
"""
import os


class AlgorithmDishonest:

    def __init__(self, algorithm):
        self.algorithm = algorithm

    def run(self, agents, cut_pattern):
        set_partitions = {}
        for idx, agent in enumerate(agents):
            print("%s running with dishonest agent %s" % (os.getpid(), idx))
            agent.setDishonesty(True)
            partitions = self.algorithm.run(agents, cut_pattern)
            set_partitions[agent.file_num] = partitions
            agent.setDishonesty(False)

        return set_partitions

    def getAlgorithmType(self):
        return "{}_{}".format("Dishonest", self.algorithm.getAlgorithmType())


if __name__ == '__main__':
    import doctest

    doctest.testmod()
