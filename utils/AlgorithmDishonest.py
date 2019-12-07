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

		# print("%s running with honest agents" % (os.getpid()))
		# partitions = self.algorithm.run(agents, cut_pattern)
		# set_partitions[len(agents)] = partitions

		return set_partitions  # todo: handle multiple partitions - this will not work like this


if __name__ == '__main__':


	import doctest
	doctest.testmod()
