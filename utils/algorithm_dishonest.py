#!python3

"""
/**
 * Implementation of the dishonest proportional cake-cutting algorithm.
 *
 * @author Itay Shtechman
 * @since 2019-04
 */
"""
import os


class AlgorithmDishonest:
    def __init__(self, cutting_algorithm):
        self.cutting_algorithm = cutting_algorithm

    def run(self, agents, cut_pattern):
        set_partitions = {}
        for idx, agent in enumerate(agents):
            print("%s running with dishonest agent %s" % (os.getpid(), idx))
            agent.set_dishonesty(True)
            partitions = self.cutting_algorithm.run(agents, cut_pattern)
            set_partitions[agent.get_map_file_number()] = partitions
            agent.set_dishonesty(False)

        return set_partitions

    def get_algorithm_type(self):
        return "{}_{}".format(
            "Dishonest", self.cutting_algorithm.get_algorithm_type()
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
