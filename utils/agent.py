#!python3
from functools import lru_cache

import numpy as np

from utils.mapfile_handler import (
    read_value_maps_from_file,
    read_value_maps_from_csv,
)
from utils.types import CutDirection


class ShadowAgent:
    """
        a shadow agent has a name and a value-function-path but does never loads valuation map from file.
        it is only used to keep runtime short when no valuations are needed and agent representation is required.
    """

    def __init__(self, value_map_path, name="Anonymous"):
        self.name = name
        self.original_name = name
        self.value_map_path = value_map_path
        self.map_file_num = self.extract_file_name(value_map_path)
        self.dishonesty = False

    def get_name(self):
        return self.name

    def get_map_file_number(self):
        return self.map_file_num

    def is_dishonest(self):
        return self.dishonesty

    def set_dishonesty(self, dishonesty):
        self.dishonesty = dishonesty
        if dishonesty:
            self.name = "Dishonest"
        else:
            self.name = self.original_name

    @staticmethod
    def extract_file_name(file_path):
        return file_path.split("/")[-1].split("_")[0]

    def get_map_path(self):
        return self.value_map_path


class Agent(ShadowAgent):
    """
    an agent has a name and a loaded value-function.
    """

    def __init__(
        self,
        value_map_path,
        name="Anonymous",
        free_play_mode=False,
        free_play_idx=-1,
    ):
        super(Agent, self).__init__(value_map_path, name)
        self.free_play_mode = free_play_mode
        self.map_file_num = (
            free_play_idx if self.free_play_mode else self.map_file_num
        )
        self.locally_loaded_value_map = None
        self.load_value_map()
        self.entire_cake_value = np.sum(self.locally_loaded_value_map)
        self.value_map_row_count = len(self.locally_loaded_value_map)
        self.value_map_col_count = len(self.locally_loaded_value_map[0])

    def load_value_map(self):
        if self.free_play_mode:
            self.locally_loaded_value_map = np.array(
                read_value_maps_from_csv(
                    self.value_map_path, self.map_file_num
                ),
                dtype=np.float,
            )
        else:
            self.locally_loaded_value_map = np.array(
                read_value_maps_from_file(self.value_map_path), dtype=np.float
            )

    def clean_memory(
        self,
    ):  # to be used for experiment with large usage of Agent
        try:
            del self.locally_loaded_value_map
        except AttributeError:
            print("passed agent %s clear", self.map_file_num)

    @lru_cache()
    def value_map_subset_sum(self, cuts_locations):
        subset = self.value_map_subset(cuts_locations)
        subset_sum = np.sum(subset)
        del subset
        return subset_sum

    @lru_cache()
    def value_map_subset(self, cuts_locations):

        i_from_row = cuts_locations[0]
        i_from_col = cuts_locations[1]
        i_to_row = cuts_locations[2]
        i_to_col = cuts_locations[3]
        if i_from_row < 0 or i_from_row > self.value_map_row_count:
            raise ValueError("i_from_row out of range: " + str(i_from_row))
        if i_from_col < 0 or i_from_col > self.value_map_col_count:
            raise ValueError("i_from_col out of range: " + str(i_from_col))
        if i_to_row < 0 or i_to_row > self.value_map_row_count:
            raise ValueError("i_to_row out of range: " + str(i_to_row))
        if i_to_col < 0 or i_to_col > self.value_map_col_count:
            raise ValueError("i_to_col out of range: " + str(i_to_col))
        if i_to_row <= i_from_row or i_to_col <= i_from_col:
            return [[]]  # special case not covered by loop below

        from_row_floor = int(np.floor(i_from_row))
        from_col_floor = int(np.floor(i_from_col))
        to_row_ceiling = int(np.ceil(i_to_row))
        to_col_ceiling = int(np.ceil(i_to_col))

        from_row_fraction = from_row_floor + 1 - i_from_row
        from_col_fraction = from_col_floor + 1 - i_from_col
        to_row_fraction = 1 - (to_row_ceiling - i_to_row)
        to_col_fraction = 1 - (to_col_ceiling - i_to_col)

        piece_value_map = self.locally_loaded_value_map[
            from_row_floor:to_row_ceiling, from_col_floor:to_col_ceiling
        ].copy()

        piece_value_map[0, :] *= from_row_fraction
        piece_value_map[-1, :] *= to_row_fraction
        piece_value_map[:, 0] *= from_col_fraction
        piece_value_map[:, -1] *= to_col_fraction
        return piece_value_map

    @lru_cache()
    def attain_row_for_desired_sum(
        self, i_from_row, i_from_col, i_to_col, wanted_sum
    ):
        self.validate_piece_indices(
            i_from_col, i_from_row, i_to_col, self.value_map_row_count
        )

        from_row_floor = int(np.floor(i_from_row))
        row_value = self.value_map_subset_sum(
            (i_from_row, i_from_col, from_row_floor + 1, i_to_col)
        )

        if row_value >= wanted_sum:
            return i_from_row + (wanted_sum / row_value)

        wanted_sum -= row_value
        for i in range(from_row_floor + 1, self.value_map_row_count):
            row_value = self.value_map_subset_sum(
                (i, i_from_col, i + 1, i_to_col)
            )
            if wanted_sum <= row_value:
                return i + (wanted_sum / row_value)
            wanted_sum -= row_value

        return self.value_map_row_count

    @lru_cache()
    def attain_col_for_desired_sum(
        self, i_from_col, i_from_row, i_to_row, wanted_sum
    ):
        self.validate_piece_indices(
            i_from_col, i_from_row, self.value_map_col_count, i_to_row
        )

        from_col_floor = int(np.floor(i_from_col))
        col_value = self.value_map_subset_sum(
            (i_from_row, i_from_col, i_to_row, from_col_floor + 1)
        )

        if col_value >= wanted_sum:
            return i_from_col + (wanted_sum / col_value)

        wanted_sum -= col_value
        for i in range(from_col_floor + 1, self.value_map_col_count):
            col_value = self.value_map_subset_sum(
                (i_from_row, i, i_to_row, i + 1)
            )
            if wanted_sum <= col_value:
                return i + (wanted_sum / col_value)
            wanted_sum -= col_value

        return self.value_map_col_count

    def validate_piece_indices(
        self, i_from_col, i_from_row, i_to_col, i_to_row
    ):
        if i_from_row < 0 or i_from_row > self.value_map_row_count:
            raise ValueError("i_from_row out of range: " + str(i_from_row))
        if i_from_col < 0 or i_from_col > self.value_map_col_count:
            raise ValueError("i_from_col out of range: " + str(i_from_col))
        if i_to_row < 0 or i_to_row > self.value_map_row_count:
            raise ValueError("i_to_row out of range: " + str(i_to_row))
        if i_to_col < 0 or i_to_col > self.value_map_col_count:
            raise ValueError("i_to_col out of range: " + str(i_to_col))
        if i_to_row <= i_from_row:
            raise ValueError("i_to_row out of range: " + str(i_to_row))
        if i_to_col <= i_from_col:
            raise ValueError("i_to_col out of range: " + str(i_to_col))

    def attain_directional_index_for_desired_sum(
        self, i_from, i_range_from, i_range_to, wanted_sum, cut_direction
    ):
        switcher = {
            CutDirection.Horizontal: self.attain_row_for_desired_sum,
            CutDirection.Vertical: self.attain_col_for_desired_sum,
        }

        def error_func():
            raise ValueError("invalid direction: " + str(cut_direction))

        inv_sum_func = switcher.get(cut_direction, error_func)
        return inv_sum_func(i_from, i_range_from, i_range_to, wanted_sum)

    @lru_cache()
    def evaluate_subset_query(self, cuts_locations):
        # self.loadValueMap()
        sum_of_subset = self.value_map_subset_sum(cuts_locations)
        # self.cleanMemory()
        return sum_of_subset

    @lru_cache()
    def mark_query_for_given_value(
        self, i_from, i_range_from, i_range_to, value, direction
    ):
        # self.loadValueMap()
        i_to = self.attain_directional_index_for_desired_sum(
            i_from, i_range_from, i_range_to, value, direction
        )
        # self.cleanMemory()
        return i_to

    @lru_cache()
    def evaluation_of_piece(self, piece):
        return self.evaluate_subset_query(piece.get_cut_indices())

    @lru_cache()
    def evaluation_of_cake(self):
        return self.entire_cake_value

    def piece_by_evaluation(self, pieces):
        """ pieces = {1: first_piece, ... , n: nth_piece} """
        sorted_piece_num_list = sorted(
            pieces.keys(),
            key=lambda piece_key: self.evaluation_of_piece(pieces[piece_key]),
            reverse=True,
        )
        return sorted_piece_num_list
