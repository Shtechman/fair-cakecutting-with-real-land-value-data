import numpy as np
class Measurements:
    @staticmethod
    def calculateAverageFaceRatio(partition):
        faceRatioList = list(map(lambda piece: piece.getFaceRatio(), partition))
        return max(0, np.average(faceRatioList))

    @staticmethod
    def calculateLargestFaceRatio(partition):
        faceRatioList = list(map(lambda piece: piece.getFaceRatio(), partition))
        return max(0, max(faceRatioList))

    @staticmethod
    def calculateSmallestFaceRatio(partition):
        faceRatioList = list(map(lambda piece: piece.getFaceRatio(), partition))
        return max(0, min(faceRatioList))

    @staticmethod
    def calculateLargestEnvy(partition):
        largestEnvyList = list(map(lambda piece: piece.getLargestEnvy(partition), partition))
        return max(1, max(largestEnvyList))

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
        relativeValues = list(map(lambda piece: max(0, piece.getRelativeValue()), partition))
        return relativeValues
