import numpy as np


class Measurements:
    @staticmethod
    def calculate_utilitarian_gain(relative_values):
        utilitarian_value = sum(relative_values)
        utilitarian_gain = utilitarian_value  # - 1
        return max(0, utilitarian_gain)

    @staticmethod
    def calculate_average_inheritance_gain(num_of_agents, relative_values):
        sell_relative_gain = 1.0 / num_of_agents
        avg_inheritance_gain = np.average(
            [rv - sell_relative_gain for rv in relative_values]
        )
        return avg_inheritance_gain

    @staticmethod
    def calculate_largest_inheritance_gain(num_of_agents, relative_values):
        sell_relative_gain = 1.0 / num_of_agents
        largest_inheritance_gain = max(
            [rv - sell_relative_gain for rv in relative_values]
        )
        return largest_inheritance_gain

    @staticmethod
    def calculate_egalitarian_gain(num_of_agents, relative_values):
        egalitarian_value = min(relative_values)
        egalitarian_gain = egalitarian_value * num_of_agents  # - 1
        return max(0, egalitarian_gain)

    @staticmethod
    def calculate_relative_values(partition):
        relative_values = {
            piece.get_agent().get_map_file_number(): max(
                0, piece.get_relative_value()
            )
            for piece in partition
        }
        return relative_values

    @staticmethod
    def calculate_absolute_values(partition):
        absolut_values = list(
            map(lambda piece: max(0, piece.get_value()), partition)
        )
        return absolut_values

    @staticmethod
    def get_egalitarian_gain(partition):
        num_of_agents = len(partition)
        relative_values = Measurements.calculate_relative_values(
            partition
        ).values()
        return Measurements.calculate_egalitarian_gain(
            num_of_agents, relative_values
        )

    @staticmethod
    def get_utilitarian_gain(partition):
        relative_values = Measurements.calculate_relative_values(
            partition
        ).values()
        return Measurements.calculate_utilitarian_gain(relative_values)

    @staticmethod
    def get_largest_envy(partition):
        largest_envy_list = list(
            map(lambda piece: piece.get_largest_envy(partition), partition)
        )
        return max(1, max(largest_envy_list))

    @staticmethod
    def get_average_face_ratio(partition):
        face_ratio_list = list(
            map(lambda piece: piece.get_face_ratio(), partition)
        )
        return max(0, np.average(face_ratio_list))

    @staticmethod
    def get_largest_face_ratio(partition):
        face_ratio_list = list(
            map(lambda piece: piece.get_face_ratio(), partition)
        )
        return max(0, max(face_ratio_list))

    @staticmethod
    def get_smallest_face_ratio(partition):
        face_ratio_list = list(
            map(lambda piece: piece.get_face_ratio(), partition)
        )
        return max(0, min(face_ratio_list))

    @staticmethod
    def merge_egalitarian_gain(
        first_eval, first_noa, second_eval, second_noa, partition
    ):
        return min(first_eval / first_noa, second_eval / second_noa) * (
            first_noa + second_noa
        )

    @staticmethod
    def merge_utilitarian_gain(
        first_eval, first_noa, second_eval, second_noa, partition
    ):
        return first_eval + second_eval

    @staticmethod
    def merge_largest_envy(
        first_eval, first_noa, second_eval, second_noa, partition
    ):
        return Measurements.get_largest_envy(partition)

    @staticmethod
    def merge_average_face_ratio(
        first_eval, first_noa, second_eval, second_noa, partition
    ):
        face_ratio_list = [first_eval] * first_noa + [second_eval] * second_noa
        return np.average(face_ratio_list)

    @staticmethod
    def merge_largest_face_ratio(
        first_eval, first_noa, second_eval, second_noa, partition
    ):
        return max(first_eval, second_eval)

    @staticmethod
    def merge_smallest_face_ratio(
        first_eval, first_noa, second_eval, second_noa, partition
    ):
        return min(first_eval, second_eval)
