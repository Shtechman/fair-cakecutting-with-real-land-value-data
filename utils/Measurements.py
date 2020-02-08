import numpy as np
class Measurements:

    @staticmethod
    def calculateUtilitarianGain(relativeValues):
        utilitarianValue = sum(relativeValues)
        utilitarianGain = utilitarianValue # - 1
        # if (utilitarianGain < -0.001): raise ValueError(
        #     "In proportional division, utilitarian gain must be at least 0; got " + str(utilitarianGain))
        return max(0, utilitarianGain)

    @staticmethod
    def calculateAverageInheritanceGain(numOfAgents, relativeValues):
        sell_relative_gain = 1.0/numOfAgents
        avgInheritanceGain = np.average([rv - sell_relative_gain for rv in relativeValues])
        return avgInheritanceGain

    @staticmethod
    def calculateLargestInheritanceGain(numOfAgents, relativeValues):
        sell_relative_gain = 1.0/numOfAgents
        largestInheritanceGain = max([rv - sell_relative_gain for rv in relativeValues])
        return largestInheritanceGain


    @staticmethod
    def calculateEgalitarianGain(numOfAgents, relativeValues):
        egalitarianValue = min(relativeValues)
        egalitarianGain = egalitarianValue * numOfAgents # - 1
        # if (egalitarianGain < -0.001): raise ValueError(
        #     "In proportional division, normalized egalitarian gain must be at least 0; got " + str(egalitarianGain));
        return max(0, egalitarianGain)

    @staticmethod
    def calculateRelativeValues(partition):
        relativeValues = {piece.getAgent().file_num: max(0, piece.getRelativeValue()) for piece in partition}
        return relativeValues

    @staticmethod
    def calculateAbsolutValues(partition):
        absolutValues = list(map(lambda piece: max(0, piece.getValue()), partition))
        return absolutValues

    @staticmethod
    def get_egalitarian_gain(partition):
        num_of_agents = len(partition)
        relative_values = Measurements.calculateRelativeValues(partition).values()
        return Measurements.calculateEgalitarianGain(num_of_agents, relative_values)

    @staticmethod
    def get_utilitarian_gain(partition):
        relative_values = Measurements.calculateRelativeValues(partition).values()
        return Measurements.calculateUtilitarianGain(relative_values)

    @staticmethod
    def get_largest_envy(partition):
        largestEnvyList = list(map(lambda piece: piece.getLargestEnvy(partition), partition))
        return max(1, max(largestEnvyList))

    @staticmethod
    def get_average_face_ratio(partition):
        faceRatioList = list(map(lambda piece: piece.getFaceRatio(), partition))
        return max(0, np.average(faceRatioList))

    @staticmethod
    def get_largest_face_ratio(partition):
        faceRatioList = list(map(lambda piece: piece.getFaceRatio(), partition))
        return max(0, max(faceRatioList))

    @staticmethod
    def get_smallest_face_ratio(partition):
        faceRatioList = list(map(lambda piece: piece.getFaceRatio(), partition))
        return max(0, min(faceRatioList))

    @staticmethod
    def merge_egalitarian_gain(first_eval,first_noa,second_eval,second_noa,partition):
        return min(first_eval/first_noa, second_eval/second_noa)*(first_noa+second_noa)

    @staticmethod
    def merge_utilitarian_gain(first_eval,first_noa,second_eval,second_noa,partition):
        return first_eval+second_eval

    @staticmethod
    def merge_largest_envy(first_eval,first_noa,second_eval,second_noa,partition):
        return Measurements.get_largest_envy(partition)

    @staticmethod
    def merge_average_face_ratio(first_eval,first_noa,second_eval,second_noa,partition):
        faceRatioList = [first_eval]*first_noa + [second_eval]*second_noa
        return np.average(faceRatioList)

    @staticmethod
    def merge_largest_face_ratio(first_eval,first_noa,second_eval,second_noa,partition):
        return max(first_eval, second_eval)

    @staticmethod
    def merge_smallest_face_ratio(first_eval,first_noa,second_eval,second_noa,partition):
        return min(first_eval,second_eval)