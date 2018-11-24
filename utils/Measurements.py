
class Measurements:
    @staticmethod
    def calculateLargestEnvy(partition):
        largestEnvyList = list(map(lambda piece: piece.getLargestEnvy(partition), partition))
        return max(largestEnvyList)

    @staticmethod
    def calculateUtilitarianGain(relativeValues):
        utilitarianValue = sum(relativeValues)
        utilitarianGain = utilitarianValue - 1;
        # if (utilitarianGain < -0.001): raise ValueError(
        #     "In proportional division, utilitarian gain must be at least 0; got " + str(utilitarianGain))
        return utilitarianGain

    @staticmethod
    def calculateEgalitarianGain(numOfAgents, relativeValues):
        egalitarianValue = min(relativeValues)
        egalitarianGain = egalitarianValue * numOfAgents - 1;
        # if (egalitarianGain < -0.001): raise ValueError(
        #     "In proportional division, normalized egalitarian gain must be at least 0; got " + str(egalitarianGain));
        return egalitarianGain

    @staticmethod
    def calculateRelativeValues(partition):
        relativeValues = list(map(lambda piece: piece.getRelativeValue(), partition))
        return relativeValues
