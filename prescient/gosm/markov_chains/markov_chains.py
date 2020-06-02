#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
markov_chains.py

This will house the machinery used to make random walks using Markov Chains.
For this purpose, it will export a MarkovChain Class which can be used
to create random walks.
"""

import datetime
from collections import OrderedDict

import numpy as np

from .markov_chains import states
from .. import sources


class MarkovError(Exception):
    pass


class MarkovChain:
    """
    This class represents a Markov Chain from which random walks can be
    generated. The user passes in a list of states, a transition mapping
    and a start state.

    This class overloads the __iter__ and __next__ methods, so it can be
    iterated over using all the list methods.

    Attributes:
        current_state: The current state of the process

    Args:
        states (List[State]): A list of states of the process, can be State
                              objects, but just needs to be a hashable object
                              with an equality operator defined.
        transition_func: One of three things:
                             1. A function from the state space to itself
                             2. A dictionary mapping states to another
                                dictionary mapping states to transition
                                probabilities.
                                {State -> {State -> probability}}
                                In this way if A is this dictionary,
                                A[S1][S2] is the probability of transitioning
                                from state S1 to state S2.
                             3. A numpy matrix of probabilities, the order
                                of rows must match order of states passed in
        start_state (State): Optional, This specifies the start state,
                             It can be a function which when called returns
                             the start state. If not passed in, the first state
                             in the list is assumed to be the start state
        end_state (State): Optional, this specifies what state to end at,
                           can be a list of states, or a function to test
                           if at an end state.
    """

    def __init__(self, states, transition_func, start_state=None,
                 end_state=None):
        self.states = states
        if callable(transition_func):
            self.transition_function = transition_func
        elif isinstance(transition_func, dict):
            self.transition_function = self._func_from_dict(transition_func)
        elif isinstance(transition_func, np.matrix):
            self.transition_function = self._func_from_matrix(transition_func)
        # current_state being None means has not gotten to start state
        self.current_state = None
        self.start_state = start_state
        self.end_state = end_state

    def _func_from_matrix(self, matrix):
        """
        Generates a transition function from a transition matrix.
        This is necessarily probabilistic. It generates a number uniformly on
        [0,1] and then iterates through the states accumulating probabilities.
        Once the accumulated probability is greater than the random number,
        iteration stops and the current state is returned.
        """
        # To find the next state,
        # We generate a random number between 0 and 1
        self.n, _ = matrix.shape
        self.matrix = matrix

        def f(state):
            random_number = np.random.rand()

            cum_prob = 0
            curr_index = self.states.index(self.current_state)
            # Then we accumulate the probabilities over the current state row
            for i in range(self.n):
                state_prob = self.matrix[curr_index,i]
                cum_prob += state_prob
                if random_number < cum_prob:
                    next_index = i
                    return self.states[next_index]
            else:
                raise MarkovError("There is no transition from state {}"
                    .format(state))

        return f

    def _func_from_dict(self, mapping):
        """
        Generates a transition function from a mapping of the form
        {State -> {State -> probability}}
        This is necessarily probabilistic. It generates a number uniformly on
        [0,1] and then iterates through the states accumulating probabilities.
        Once the accumulated probability is greater than the random number,
        iteration stops and the current state is returned.
        """
        self.mapping = mapping
        def f(state):
            # To find the next state,
            # We generate a random number between 0 and 1
            random_number = np.random.rand()

            cum_prob = 0

            # Then we accumulate the probabilities over the current state row
            for state in self.mapping[self.current_state]:
                state_probability = self.mapping[self.current_state][state]
                cum_prob += state_probability
                if random_number < cum_prob:
                    return state
            else:
                raise MarkovError("There is no transition from state {}"
                    .format(state))
        return f

    def _pick_start_state(self):
        """
        Sets the current state attribute to the start state
        """
        start_state = self.start_state

        if start_state is None:
            # If no start_state passed in, we assume first state is start
            self.current_state = self.states[0]
        elif callable(start_state):
            # If start_state is a function, we call it to get first state
            self.current_state = start_state()
        else:
            # Otherwise, what is passed in is a state
            self.current_state = start_state

    def step(self):
        """
        Moves the current state attribute to the next state according to
        the function. Picks a start state if current state is None.
        """
        if self.current_state is None:
            self._pick_start_state()
        else:
            if self.is_end_state(self.current_state):
                raise StopIteration
            self.current_state = self.transition_function(self.current_state)

    def is_end_state(self, state):
        """
        This function checks if the passed in state is an end state.
        """
        if self.end_state is None:
            return False
        elif isinstance(self.end_state, states.State):
            return state == self.end_state
        elif isinstance(self.end_state, list):
            return state in self.end_state
        elif callable(self.end_state):
            return self.end_state(state)
        else:
            return state == self.end_state

    def reset(self):
        """
        Sets the current state attribute to None, so the Markov Chain
        can be run again.
        """
        self.current_state = None

    def random_walk(self, n=None):
        """
        This generates states and yields them (so it does not create a list
        in memory). This should not be called in conjunction with other methods
        which change the current state as then the walk generated may not
        correspond to an actual walk.

        This resets the current state whenever it is called.

        Args:
            n (int): The number of steps taken in the random walk,
                     not passed in if want to generate states indefinitely
        """
        self.reset()
        if n is None:
            while True:
                self.step()
                yield self.current_state
        else:
            for _ in range(n):
                self.step()
                yield self.current_state

    def __iter__(self):
        return self

    def __next__(self):
        self.step()
        return self.current_state


def increment(mapping, state, next_state, count=1):
    """
    Increments mapping[state][next_state], creating entries
    if need be

    Specify count you want to increment by something other
    than 1.

    Args:
        mapping (dict[State,dict[State,int]]): The transition counts
        state (State): The state of first state
        next_state (State): The state to transition to
        count (int): The amount to increment by
    """
    if state in mapping:
        if next_state in mapping[state]:
            mapping[state][next_state] += count
        else:
            mapping[state][next_state] = count
    else:
        mapping[state] = {next_state:count}


def mapping_from_walk(walk, memory=1):
    """
    From a sequence of states, generates the count map from which one
    state transitions to another. This is stored as a dictionary of the form
    {State -> {State -> count}}.

    The memory argument means to consider the frequency n states transition
    to the next n states. For example, if we have states [A, B, A, A] and
    set memory = 2,
    our mapping would be {(A, B): {(B, A): 1}, (B, A): {(A, A): 1}}

    If memory > 1, the keys in the dictionary will be tuples of states

    Args:
        walk (List[State]): A sequence of states
        memory (int): A number representing how many states to use for memory
    """

    count_map = {}


    if memory == 1:
        for (state, next_state) in zip(walk, walk[1:]):
            increment(count_map, state, next_state)
    else:
        offsets = [walk[i:] for i in range(memory)]
        state_tuples = list(zip(*offsets))
        for (state, next_state) in zip(state_tuples, state_tuples[1:]):
            increment(count_map, state, next_state)

    return count_map


def merge_maps(maps):
    """
    Merges counts of transitions from multiple different mappings
    into a single mapping.

    Args:
        maps (List[dict]): A list of mappings, these should be dictionaries
            of States mapped to dictionaries of States mapped to numbers
            {State -> {State -> int}}
    Returns:
        dict: A dictionary of the merged counts
    """
    merged = {}
    for mapping in maps:
        for state in mapping:
            for next_state in mapping[state]:
                count = mapping[state][next_state]
                increment(merged, state, next_state, count)
    return merged


def normalize_map(mapping):
    """
    Creates a new dictionary with the frequency of each transition.
    Each state transition count is normalized by the total number of
    transitions out of a given state.

    Args:
        maps (List[dict]): A list of mappings, these should be dictionaries
            of States mapped to dictionaries of States mapped to numbers
            {State -> {State -> int}}
    Returns:
        dict: A dictionary of the normalized counts
    """
    normalized_dict = {}
    for word in mapping:
        normalized_dict[word] = {}
        count = sum(mapping[word].values())
        for other in mapping[word]:
            normalized_dict[word][other] = mapping[word][other] / count
    return normalized_dict


def matrix_from_mapping(mapping):
    """
    From a mapping of the form {State -> {State -> probability}}, it creates
    an equivalent transition matrix. Note if the mapping is just a dictionary,
    not an OrderedDict, the ordering of the rows on the matrix will be
    nondeterministic.

    Args:
        mapping (dict): A dictionary mapping pairs of states to probabilities
    Returns:
        (np.Matrix): A numpy transition matrix
    """

    states = list(mapping)
    n = len(states)

    matrix = np.matrix(np.zeros((n,n)))

    for i, state in enumerate(states):
        for j, next_state in enumerate(states):
            matrix[i,j] = mapping[state].get(next_state, 0)
    return matrix


def generate_equal_probability_matrix(n):
    """
    Generates a nxn transition matrix where the probability
    of any given transition is the same
    Args:
        n (int): The dimensions of the matrix
    Returns:
        np.matrix: The matrix desired
    """

    num_array = np.ones((n,n)) * (1/n)
    return np.matrix(num_array)


def sort_dictionary(dictionary):
    """
    This function will turn a dictionary with sortable keys into an
    OrderedDict with keys in the ordering of the keys.

    This dictionary will have two levels of nesting as it will be a mapping of
    the sort {State -> {State -> value}}

    Args:
        dictionary (dict): The dictionary to be sorted
    Returns:
        OrderedDict: The sorted dictionary
    """
    ordered = OrderedDict()
    for outer_key in sorted(dictionary):
        inner_ordered = OrderedDict()
        for inner_key in sorted(dictionary[outer_key]):
            inner_ordered[inner_key] = dictionary[outer_key][inner_key]
        ordered[outer_key] = inner_ordered
    return ordered


def matrix_from_walk(state_walk, memory=1):
    """
    This function will construct a transition matrix from a sequence of states
    encoded in an OrderedDict. The state_walk is an OrderedDict which maps
    datetimes to States, and this function will count how many times between
    consecutive hours, one state switches to another. From this, we compute
    the frequency of each transition and compute the matrix.

    In addition, we can consider transitions from combinations of states to
    other combinations of states. How many states is specified by the memory
    argument. For example if we had the states [A, B, C] with memory=2,
    we would record one transition from (A, B) to (B, C).

    This will attempt to sort the states.

    Args:
        state_walk (OrderedDict[datetime,State]): A mapping from datetimes to
            states
        memory (int): How many states to consider as a single state.

    Returns:
        (List[State], np.matrix): The order of the states and the matrix
    """

    segments = create_consecutive_segments(state_walk)
    mappings = [mapping_from_walk(list(segment.values()), memory)
                for segment in segments]
    merged = merge_maps(mappings)
    normalized_map = normalize_map(merged)

    final_mapping = sort_dictionary(normalized_map)

    return list(final_mapping), matrix_from_mapping(final_mapping)


def date_range(start, end):
    """
    Generates all datetimes between the start and end date.
    This function should work with any datetime object which supports
    addition with datetime.timedelta. Datetimes are separated by one hour each.

    Args:
        start (datetime-like): The start datetime
        end (datetime-like): The end datetime
    """
    current_datetime = start
    while current_datetime != end:
        yield current_datetime
        current_datetime += datetime.timedelta(hours=1)
    yield current_datetime


def create_consecutive_segments(date_indexed_dict):
    """
    This function breaks up a datetime indexed dictionary into the consecutive
    segments of datetimes. While iterating through the datetimes, if it finds
    any hourly gap in the data, it breaks the dictionary into two dictionaries.
    At the end, it will have a list of dictionaries, where each dictionary
    has no gaps.

    Args:
        date_indexed_dict (dict[datetime,State]): A datetime indexed dictionary
    Returns:
        List[dict]: List of dictionaries for which the indices are consecutive
            and are separated by one hour
    """
    min_date = min(date_indexed_dict)
    max_date = max(date_indexed_dict)

    segments = []

    # We have a flag for when we need to start a new dictionary
    create_new = True
    for dt in date_range(min_date, max_date):
        if create_new:
            current_dictionary = OrderedDict()
            segments.append(current_dictionary)
            create_new = False

        # We create a new dictionary whenever we reach a datetime which is
        # not in the original dictionary, but the next datetime is
        if (dt not in date_indexed_dict and
            dt+datetime.timedelta(hours=1) in date_indexed_dict):
            create_new = True
        else:
            if dt in date_indexed_dict:
                current_dictionary[dt] = date_indexed_dict[dt]
    return segments
