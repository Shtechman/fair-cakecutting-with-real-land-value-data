#!python3

"""
/**
 * @author Itay Shtechman
 * @since 2018-11
 */
"""
from functools import lru_cache

from utils.types import CutDirection


class Piece:
    """
    A class representing a piece of 2-dimensional cake.

    @author Itay Shtechman
    @since 2018-11
    """

    def __init__(self, i_from_row, i_from_col, i_to_row, i_to_col):
        """ /**
         *  Initialize a 2-dimensional piece.
         *  @param i_from_row float location of the start vertical cut
         *  @param i_from_col float location of the start horizontal cut
         *  @param i_to_row float location of the end vertical cut
         *  @param i_to_col float location of the end horizontal cut
         */ """
        self.i_from_row = i_from_row
        self.i_from_col = i_from_col
        self.i_to_row = i_to_row
        self.i_to_col = i_to_col

    def __repr__(self):
        return "[%0.2f,%0.2f,%0.2f,%0.2f] - (Ratio %0.2f)" % (
            self.i_from_row,
            self.i_from_col,
            self.i_to_row,
            self.i_to_col,
            self.get_face_ratio(),
        )

    def to_string(self):
        return self.__repr__()

    def get_allocated_piece(self, agent):
        return AllocatedPiece(
            agent,
            self.i_from_row,
            self.i_from_col,
            self.i_to_row,
            self.i_to_col,
        )

    def get_opposite_directional_indices(self, direction):
        switcher = {
            CutDirection.Horizontal: (self.i_from_col, self.i_to_col),
            CutDirection.Vertical: (self.i_from_row, self.i_to_row),
        }

        return switcher.get(direction, None)

    def get_directional_i_from(self, direction):
        switcher = {
            CutDirection.Horizontal: self.i_from_row,
            CutDirection.Vertical: self.i_from_col,
        }

        return switcher.get(direction, None)

    def get_directional_i_to(self, direction):
        switcher = {
            CutDirection.Horizontal: self.i_to_row,
            CutDirection.Vertical: self.i_to_col,
        }

        return switcher.get(direction, None)

    def get_dimensions(self):
        return {
            CutDirection.Horizontal: self.i_to_row - self.i_from_row,
            CutDirection.Vertical: self.i_to_col - self.i_from_col,
        }

    def get_i_from_col(self):
        return self.i_from_col

    def get_i_to_col(self):
        return self.i_to_col

    def get_i_from_row(self):
        return self.i_from_row

    def get_i_to_row(self):
        return self.i_to_row

    def get_cut_indices(self):
        return self.i_from_row, self.i_from_col, self.i_to_row, self.i_to_col

    def get_face_ratio(self):
        width = self.i_to_col - self.i_from_col
        height = self.i_to_row - self.i_from_row
        return min(width, height) / max(width, height)


class AllocatedPiece(Piece):
    """
    A class representing a piece allocated to an agent on a 2-dimensional cake.

    @author Itay Shtechman
    @since 2018-11
    """

    def __init__(
        self,
        agent,
        i_from_row=None,
        i_from_col=None,
        i_to_row=None,
        i_to_col=None,
    ):
        """ /**
         *  Initialize a 2-dimensional allocated piece based on a value function.
         *  @param agent an Agent.
         *  @param i_from_row float location of the start vertical cut
         *  @param i_from_col float location of the start horizontal cut
         *  @param i_to_row float location of the end vertical cut
         *  @param i_to_col float location of the end horizontal cut
         */ """
        if i_from_row is None:
            i_from_row = 0
        if i_from_col is None:
            i_from_col = 0
        if i_to_row is None:
            i_to_row = agent.value_map_row_count
        if i_to_col is None:
            i_to_col = agent.value_map_col_count

        super(AllocatedPiece, self).__init__(
            i_from_row, i_from_col, i_to_row, i_to_col
        )
        self.agent = agent
        self.agent_name = agent.name
        self.cut_marks = {}

    def __copy__(self):
        copy_piece = AllocatedPiece(
            self.agent,
            self.i_from_row,
            self.i_from_col,
            self.i_to_row,
            self.i_to_col,
        )
        copy_piece.cut_marks = self.cut_marks
        return copy_piece

    def __repr__(self):
        return "%s(%s) receives %s" % (
            self.agent_name,
            self.agent.get_map_file_number(),
            super(AllocatedPiece, self).__repr__(),
        )

    def to_string(self):
        return self.__repr__()

    def clear(self):
        self.agent.clean_memory()
        del self.agent

    def subcut(self, i_dir_from, i_dir_to, direction):
        switcher = {
            CutDirection.Horizontal: (
                i_dir_from,
                self.i_from_col,
                i_dir_to,
                self.i_to_col,
            ),
            CutDirection.Vertical: (
                self.i_from_row,
                i_dir_from,
                self.i_to_row,
                i_dir_to,
            ),
        }

        i_from_row, i_from_col, i_to_row, i_to_col = switcher.get(
            direction, None
        )

        return AllocatedPiece(
            self.agent, i_from_row, i_from_col, i_to_row, i_to_col
        )

    def get_agent(self):
        return self.agent

    def get_directional_value(self, cut_location, direction):
        """ Given a direction and a cut query, calculate what would be the piece relative value """
        if self.cut_marks[direction] < cut_location:
            subcut_i_from = self.get_directional_i_from(direction)
            subcut_i_to = cut_location
        else:
            subcut_i_from = cut_location
            subcut_i_to = self.get_directional_i_to(direction)
        return self.subcut(
            subcut_i_from, subcut_i_to, direction
        ).get_relative_value()

    def get_directional_face_ratio(self, cut_location, direction):
        """ Given a direction and a cut query, calculate what would be the piece face ratio """
        if self.cut_marks[direction] < cut_location:
            subcut_iFrom = self.get_directional_i_from(direction)
            subcut_iTo = cut_location
        else:
            subcut_iFrom = cut_location
            subcut_iTo = self.get_directional_i_to(direction)
        return self.subcut(
            subcut_iFrom, subcut_iTo, direction
        ).get_face_ratio()

    def get_value(self):
        """
        The current agent evaluates his own piece
        """
        return self.agent.evaluation_of_piece(self)

    def get_relative_value(self):
        """
        The current agent evaluates his own piece relative to the entire cake
        """
        return (
            self.agent.evaluation_of_piece(self)
            / self.agent.evaluation_of_cake()
        )

    def get_envy(self, other_piece):
        """
        The current agent reports his relative envy of the other agent's piece
        """
        envious_value = self.agent.evaluation_of_piece(self)
        envied_value = self.agent.evaluation_of_piece(other_piece)
        if envious_value >= envied_value:
            return 1
        else:
            return envied_value / envious_value

    def get_largest_envy(self, other_pieces):
        """
        The current agent reports his largest relative envy of another agent's piece
        """

        def get_envy(piece):
            return self.get_envy(piece)

        return max(
            list(map(lambda other_piece: get_envy(other_piece), other_pieces))
        )

    @lru_cache()
    def mark_query_for_given_value(self, value, cut_direction):
        switcher = {
            CutDirection.Horizontal: self.mark_horizontal_query_for_given_value,
            CutDirection.Vertical: self.mark_vertical_query_for_given_value,
        }

        def error_func():
            raise ValueError("invalid direction: " + str(cut_direction))

        mark_query_func = switcher.get(cut_direction, error_func)
        return mark_query_func(value)

    @lru_cache()
    def mark_horizontal_query_for_given_value(self, value):
        return self.agent.mark_query_for_given_value(
            self.i_from_row,
            self.i_from_col,
            self.i_to_col,
            value,
            CutDirection.Horizontal,
        )

    @lru_cache()
    def mark_vertical_query_for_given_value(self, value):
        return self.agent.mark_query_for_given_value(
            self.i_from_col,
            self.i_from_row,
            self.i_to_row,
            value,
            CutDirection.Vertical,
        )
