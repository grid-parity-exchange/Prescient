#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

# -----------------------------------------------------------------------
# The following classes mirror the structure of possible hyperrectangles.
# -----------------------------------------------------------------------

from itertools import combinations, product
try:
    from math import gcd
except ImportError:
    from fractions import gcd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from prescient.gosm.multidim_utils import MultiDimMixIn

def parse_pattern(lines, dimension_names):
    """
    This function is used for the hyperrectangle file parsers. It will take
    a list of the lines for a specific pattern and the dimension names
    associated with the pattern and construct a HyperrectanglePattern.

    Args:
        lines (list[str]): The list of lines with the pattern
        dimension_names (list[str]): The list of dimension names
    Returns:
        HyperrectanglePattern: The parsed out hyperrectangle pattern
            constructed using the dimension names
    """
    hyperrectangles = []
    for line in lines:
        if line.startswith('Pattern'):
            try:
                name = line.split()[1]
            except IndexError:
                raise RuntimeError('You must provide a pattern name for '
                                   'each pattern.')
        if line.startswith('-'):
            rect_name, *bound_strings = line[1:].split()
            bounds = []
            for bound_string in bound_strings:
                # Drop outer parenthesis
                inner = bound_string[1:-1].split(',')
                lower, upper = float(inner[0]), float(inner[1])
                bounds.append((lower, upper))
            rect = Hyperrectangle(bounds, dimension_names, rect_name)
            hyperrectangles.append(rect)
    return HyperrectanglePattern(name, hyperrectangles, dimension_names)


def one_dimensional_pattern_set_from_file(filename, dimension_name=0):
    """
    This function will parse out one-dimensional hyperrectangles (intervals)
    from the passed in filename with the optionally specified dimension name.
    The pattern set will contain Interval objects.

    Args:
        filename (str): The file containing the hyperrectangles
        dimension_name (str): The name of the dimension associated with the
            interval, defaults to 0 if not passed
    Returns:
        HyperrectanglePatternSet: The set of Hyperrectangle Patterns
    """
    with open(filename) as input_file:
        # Read lines without '\n' and close the input file.
        lines = input_file.read().splitlines()

    # Find the dimension keys and all line indices where the
    # definitions of patterns start.
    pattern_line_indices = []
    for index, line in enumerate(lines):
        if line.startswith('Pattern'):
            pattern_line_indices.append(index)

    # Check if at least one pattern was defined.
    if not len(pattern_line_indices):
        raise RuntimeError('You must define at least one pattern.')

    # Append index of the last line + 1 for the following loop.
    pattern_line_indices.append(len(lines))

    dimkeys = [dimension_name] if dimension_name is not None else None
    patterns = {}
    # Iterate over all lines in which the definition of a pattern starts.
    for ix, next_ix in zip(pattern_line_indices, pattern_line_indices[1:]):
        intervals = []
        for line in lines[ix:next_ix]:
            if line.startswith('Pattern'):
                try:
                    name = line.split()[1]
                except IndexError:
                    raise RuntimeError('You must provide a pattern name for '
                                       'each pattern.')
            if line.startswith('-'):
                interval_name, *bound_strings = line[1:].split()
                bounds = []
                for bound_string in bound_strings:
                    # Drop outer parenthesis
                    inner = bound_string[1:-1].split(',')
                    lower, upper = float(inner[0]), float(inner[1])
                    bounds.append((lower, upper))
                if len(bounds) > 1:
                    raise RuntimeError("This function only expects one dimensional patterns")
                a, b = bounds[0]
                interval = Interval(a, b, interval_name, dimension_name)
                intervals.append(interval)
        pattern = HyperrectanglePattern(name, intervals, dimkeys)
        patterns[pattern.name] = pattern

    return HyperrectanglePatternSet(patterns, dimkeys=dimkeys)


def multi_dimensional_pattern_set_from_file(filename):
    """
    This should construct a HyperrectanglePatternSet from a file which is
    structured as specified in the following notes

    Notes:
        How the files containing the hyperrectangles' names has to be
        structured:
        - The first nonempty line has to contain the keyword "Sources",
          followed by a whitespace and the names of the dimkeys, separated
          by whitespaces. If the defined hyperrectangles are
          one-dimensional, you don't have to specify the sources.
          In this case, there should be no keyword "Sources".
        - Patterns start with the keyword "Pattern" followed by a colon,
          a whitespace and the name of the pattern.
        - The hyperrectangles belonging to this pattern must be listed
          below, each hyperrectangle in a new line starting with "-",
          followed by the hyperrectangle's name (cannot be
          "<pattern_name>_residual"), a whitespace and the intervals in the
          order of the dimkeys as defined above, separated by whitespaces.
        - An interval must be written as "(x,y)", where "x" defines the
          lower and "y" the upper bound. The bounds cannot have more than
          four decimal places.
        - Every hyperrectangle of one pattern must be either disjoint to
          all other hyperrectangles of this pattern, or a proper subset of
          an other hyperrectangle. If a hyperrectangle contains a proper
          subset, the subset is subtracted from the hyperrectangle.
        - The whole unit cube has to be covered by hyperrectangles.
          The easiest way to cover residual space is to
          create one hyperrectangle with every interval equal to (0,1).

    Examples:
        Sources: SoCalSolar SoCalWind

        Pattern: sunriseuni3
        -sunu3low (0,1) (0,0.33)
        -sunu3mid (0,1) (0.33,0.67)
        -sunu3high (0,1) (0.67,1)

        Pattern: wide1
        -widelow (0,0.1) (0,0.1)
        -widehigh (0.9,1) (0.9,1)

    This will return a corresponding HyperrectanglePatternSet with a
    hyperrectangle pattern set for each one that is defined in the file.

    Args:
        filename (str): The path to the file containing the hyperrectangles
    Returns:
        HyperrectanglePatternSet: The corresponding pattern set
    """
    with open(filename) as input_file:
        # Read lines without '\n' and close the input file.
        lines = input_file.read().splitlines()

    # Temporarily save the names of the dimkeys.
    dimkeys = []

    dimkeys_found = False

    # Find the dimension keys and all line indices where the
    # definitions of patterns start.
    pattern_line_indices = []
    for index, line in enumerate(lines):
        if line.startswith('Pattern'):
            if dimkeys_found:
                pattern_line_indices.append(index)
            elif len(dimension_names) > 1:
                raise RuntimeError('You must define the dimkeys'
                                   'before defining any pattern.')
        elif line.startswith('Sources') and not dimkeys_found:
            dimkeys = line.split()[1:]
            dimkeys_found = True
    if not dimkeys_found:
        raise RuntimeError("Couldn't find a line starting"
                           "with 'Sources' in file '{}'.".format(filename))

    # Check if at least one dimension was defined.
    if len(dimkeys) == 0:
        raise RuntimeError("You must define at least one"
                           "source in file '{}'.".format(filename))

    # Check if at least one pattern was defined.
    if not len(pattern_line_indices):
        raise RuntimeError('You must define at least one pattern.')

    # Append index of the last line + 1 for the following loop.
    pattern_line_indices.append(len(lines))

    patterns = {}
    # Iterate over all lines in which the definition of a pattern starts.
    for ix, next_ix in zip(pattern_line_indices, pattern_line_indices[1:]):
        pattern = parse_pattern(lines[ix:next_ix])
        patterns[pattern.name] = pattern

    return HyperrectanglePatternSet(patterns)


def interval_is_subset(bounds1, bounds2):
    """
    Checks if the interval specified in bounds1 is a subset of the interval
    specified in bounds2. If this is not the case but the intervals are not
    disjoint either, an error is thrown.

    Args:
        bounds1: the interval which is checked to being a subset of bounds2
        bounds2: the interval which is checked to being a superset of
            bounds1

    Returns:
        tuple: tuple (x,y,z) with x=True, if the interval specified in
               bounds1 is a subset (y=True if proper subset) of the
               interval specified in bounds2, x=False otherwise;
               z=True if the intervals are disjoint, False otherwise.
    """

    is_subset = False
    is_proper_subset = False
    are_disjoint = False

    # Check if bounds1 is a subset of bounds2.
    if bounds1[0] >= bounds2[0] and bounds1[1] <= bounds2[1]:
        is_subset = True
        # Check if the subset is proper.
        if bounds1[0] > bounds2[0] or bounds1[1] < bounds2[1]:
            is_proper_subset = True
    # If bounds1 is not a subset of bounds2,
    # check if the intervals are disjoint.
    elif bounds1[1] <= bounds2[0] or bounds1[0] >= bounds2[1]:
        are_disjoint = True

    return is_subset, is_proper_subset, are_disjoint


class HyperrectanglePatternSet(MultiDimMixIn):
    """
    This class manages all possible patterns of hyperrectangles.

    Attributes:
        patterns (dict[str, HyperrectanglePattern]): a dictionary containing
            all patterns accessible by their names
    """
    def __init__(self, patterns, dimkeys=None):
        """
        Initializes an instance of the HyperrectanglePatternSet class and reads
        the data.

        Args:
            dict[str,HyperrectanglePattern]: A dictionary mapping name of
                hyperrectangle patterns to HyperrectanglePattern
        """
        # A dictionary like {pattern_name -> HyperrectanglePattern}.
        self.patterns = patterns

        sample_pattern = list(patterns.values())[0]
        MultiDimMixIn.__init__(self, sample_pattern.ndim, dimkeys)

        for pattern in patterns.values():
            if not self.match(pattern):
                raise ValueError("Not all patterns passed in have matching "
                                 "dimension names.")

        self._check_uniqueness()

    def _check_uniqueness(self):
        """
        Checks if the defined hyperrectangle names are unique.

        Args:
            dimkeys (list): list of dimkeys
        """

        # Check if the hyperrectangle names are unique.
        names = []
        for pattern in self.patterns.values():
            for rect in pattern.hyperrectangles:
                if rect.name in names:
                    raise ValueError("The hyperrectangle name '{}' "
                                     "is not unique.".format(rect_name))
                else:
                    names.append(rect.name)

    def get_pattern(self, name):
        """
        Return the HyperrectanglePattern with the name passed in

        Args:
            name (str): The name of the pattern
        Returns:
            HyperrectanglePattern: The associated pattern
        """
        for pattern_name, pattern in self.patterns.items():
            if name == pattern_name:
                return pattern
        else:
            raise RuntimeError("No pattern with name {} found.".format(name))

    def get_hyperrectangle(self, name):
        """
        Return the hyperrectangle with the given name.

        Args:
            name (str): the name of the hyperrectangle to return

        Returns:
            Hyperrectangle: the hyperrectangle
        """

        for pattern in self.patterns.values():
            for hyperrectangle in pattern.hyperrectangles:
                if hyperrectangle.name == name:
                    return hyperrectangle

        raise RuntimeError("No hyperrectangle named '{}' found.".format(name))

    def set_dimkeys(self, dimkeys):
        """
        This will set the names of the dimensions for the pattern set and
        every pattern contained in the set.

        Args:
            dimkeys (list): The names of the dimensions
        """
        self.dimkeys = dimkeys
        for pattern in self.patterns.values():
            pattern.set_dimkeys(dimkeys)


class HyperrectanglePattern(MultiDimMixIn):
    """
    This class manages one possible pattern of hyperrectangles.

    Attributes:
        name (str): the name of the hyperrectangle
        dimkeys (list[str]): a list of the names corresponding to the
            dimensions
        hyperrectangles (set[HyperrectangleWithCutouts]): a set of
            HyperrectangleWithCutouts contained in this pattern
    """

    def __init__(self, name, hyperrectangles, dimkeys=None):
        """
        Initializes an instance of the HyperrectanglePattern class.

        Args:
            name (str): the name of the pattern
            hyperrectangles (list[Hyperrectangle]): A list of Hyperrectangles
                which compose the patterns
            dimkeys (list[str]): Optionally, the names of the dimensions of
                the rectangles
        """

        self.name = name
        # We store internally HyperrectangleWithCutouts objects to account
        # for the possibility that some rectangles might be cut out of others
        self.hyperrectangles = {rect.add_cutouts() for rect in hyperrectangles}

        if len(hyperrectangles) == 0:
            raise ValueError("A HyperrectanglePattern must contain at least "
                             "one hyperrectangle.")
        one_rect = hyperrectangles[0]

        MultiDimMixIn.__init__(self, one_rect.ndim, dimkeys)

        for rect in hyperrectangles:
            if not self.match(rect):
                raise ValueError("All hyperrectangles in a pattern must have "
                                 "the same dimkeys.")

        # Update the cutout set of each hyperrectangle.
        self._handle_proper_subsets()

        # Check if there is any uncovered residual space.
        self._check_for_residual_space()

    def _check_for_residual_space(self):
        """
        Check if there is any residual space uncovered by the user defined
        hyperrectangles. It does this by checking if the volume of the
        hyperrectangles contained in this pattern sums to 1.

        This is valid if each hyperrectangle is contained within [0,1]^n and
        are disjoint.
        """
        covered_volume = sum(rect.volume for rect in self.hyperrectangles)
        if not np.isclose(covered_volume,  1):
            raise ValueError("The passed in rectangles do not cover "
                             "all of [0,1]^n")

    def _handle_proper_subsets(self):
        """
        Checks for overlapping hyperrectangles and adds every proper subset of
        each hyperrectangle to its cutout list and deletes all nested cutouts.
        If two overlapping hyperrectangles are found, that are neither disjoint
        nor is one of them a subset of the other, an error is thrown.
        """
        # Iterate over all 2-combinations of hyperrectangles.
        for i, rect1 in enumerate(self.hyperrectangles):
            # We count the number of rectangles this is a subset of
            # If this a subset of 2 or more rectangles, we raise an error
            sub_count = 0
            for j, rect2 in enumerate(self.hyperrectangles):
                if i == j:
                    continue
                # Get the relation between rect1 and rect2.
                relation = rect1.rectangle.get_relation(rect2.rectangle)
                if relation == 'proper_super':
                    rect2.cutouts.add(rect1.rectangle)
                    sub_count += 1
                elif relation == 'equal':
                    raise ValueError("The hyperrectangles '{}' and '{}' of "
                                     "pattern '{}' are the same."
                                     .format(rect1.name, rect2.name, self.name))
                elif relation == 'neither':
                    raise ValueError("The hyperrectangles '{}' and '{}' of "
                                     "pattern '{}' are neither disjoint nor is "
                                     "one of them a subset of the other."
                                     .format(rect1.name, rect2.name, self.name))
            if sub_count > 1:
                raise ValueError("The hyperrectangle '{}' is a subset of 2 or "
                                 "more hyperrectangles.".format(rect1.name))

    def set_dimkeys(self, dimkeys):
        """
        This will set the names of the dimensions for all hyperrectangles
        contained in this pattern.

        Args:
            dimkeys (list): The names of the dimensions
        """
        self.dimkeys = dimkeys
        for rect in self.hyperrectangles:
            rect.set_dimkeys(dimkeys)

    def __str__(self):
        string = self.name + '\n'
        for rect in self.hyperrectangles:
            string += str(rect) + '\n'
        return string

    def __repr__(self):
        return "HyperrectanglePattern({})".format(self.name)


class Hyperrectangle(MultiDimMixIn):
    """
    This class manages one possible hyperrectangle. A hyperrectangle with a
    value other than None is also referred to as a skeleton point.

    Attributes:
        name (str): the name of the hyperrectangle
        dimkeys (list[str]): a list of the dimension names
        bounds (dict[str, dict[str, float]]): a dictionary to save the lower
            and upper bounds of the intervals for each dimension
    """

    def __init__(self, bounds, dimkeys=None, name=None):
        """
        Initializes an instance of the Hyperrectangle class.

        Args:
            bounds: Either a list containing one interval for each dimension,
                or a dictionary accessible through dimension names
            dimkeys (list[str]): Optionally, a list of the dimension names,
                if None passed in it will default to integers 0,1,2,...,n-1.
            name (str): The name of the hyperrectangle
        """

        self.name = name

        ndim = len(bounds)
        MultiDimMixIn.__init__(self, ndim, dimkeys)

        # A nested dictionary like {dimension -> tuple}
        # where tuple is an interval
        self.bounds = {}

        # If bounds is a list, it contains strings describing the
        # intervals.
        if isinstance(bounds, (list, tuple)):
            if len(bounds) < len(self.dimkeys):
                raise ValueError('You must provide an interval for each '
                                 "dimension for hyperrectangle "
                                 "'{}'.".format(self.name))
            elif len(bounds) > len(self.dimkeys):
                print("Warning: You provided too many intervals for"
                      " hyperrectangle '{}'. Only the first {} intervals"
                      " are being used.".format(name, len(self.dimkeys)))

            for dimension, bounds in zip(self.dimkeys, bounds):
                if len(bounds) != 2:
                    raise ValueError("Each dimension's bounding interval must "
                                     "have exactly two values.")
                self.bounds[dimension] = bounds

        # If bounds is not a list, it is a dictionary and already has
        # the right structure. We only need to round the interval bounds to
        # 4 decimal places (there might be more decimal places due to
        # binary approximation of the computer).
        elif isinstance(bounds, dict):
            for dimension in self.dimkeys:
                self.bounds[dimension] = tuple([round(bound, 4)
                                             for bound in bounds[dimension]])

        self._check_consistency()

    def _check_consistency(self):
        """
        Checks if the defined hyperrectangles are consistent and
        obey specific rules.
        """

        for dimension, bounds in self.bounds.items():

            # Check if the lower bound is greater or equal 0 and the upper
            # bound less or equal 1.
            lower, upper = bounds
            if lower < 0 or upper > 1:
                raise RuntimeError("The lower and upper bounds defined for "
                                   "hyperrectangle '{}' must lie in "
                                   "the interval [0,1].".format(self.name))

            # Check if the upper bound is in fact greater than the lower bound.
            if lower >= upper:
                raise RuntimeError("The upper bound of an interval is less or "
                                   "equal its lower bound. Check rectangle"
                                   " '{}', dimension '{}'.".format(self.name,
                                                                dimension))

    def create_union(self, rect, dimension):
        """
        Creates a union of two hyperrectangles (self and rect) along the axis
        specified by souce if the union is a hyperrectangle itself.
        If a union cannot be made, this raises a ValueError.

        Args:
            rect (Hyperrectangle): the hyperrectangle to unify self with
            dimension (str): the name of the dimension along which the
                union is to be created
        Returns:
            Hyperrectangle: the union of self and rect (if it is a
                hyperrectangle)
        """

        if not self.match(rect):
            raise ValueError("The dimensions of the the two rectangles don't "
                             "match.")

        # Save the intervals for the union that might be created.
        new_bounds = {}

        # Check if all intervals except the one belonging to dimension
        # are the same.
        for dim in self.dimkeys:
            if dim == dimension:
                continue
            new_bounds[dim] = self.bounds[dim]
            # If the interval is not the same, we raise an error
            if self.bounds[dim] != rect.bounds[dim]:
                raise ValueError("The rectangles must match on every other "
                                 "axis then the one specified.")

        # If this part of the code is reached, we are sure, that all
        # intervals except the one belonging to dimension are the same.
        # Now compare the intervals belonging to dimension.
        if self.bounds[dimension][1] == rect.bounds[dimension][0]:
            # self is adjacent to the left of rect.
            new_bounds[dimension] = (self.bounds[dimension][0],
                                     rect.bounds[dimension][1])
            return Hyperrectangle(bounds=new_bounds, dimkeys=self.dimkeys)
        elif self.bounds[dimension][0] == rect.bounds[dimension][1]:
            # self is adjacent to the right of rect.
            new_bounds[dimension] = (rect.bounds[dimension][0],
                                     self.bounds[dimension][1])
            return Hyperrectangle(bounds=new_bounds, dimkeys=self.dimkeys)
        else:
            raise ValueError("Rectangles are not adjacent on the axis "
                             "specified.")

    def compute_volume(self):
        """
        Computes the volume of the hyperrectangle. This is needed for the
        probability in the case of independence across day part separators.

        Returns:
            float: the volume
        """

        volume = 1
        for dim in self.dimkeys:
            volume *= self.bounds[dim][1] - self.bounds[dim][0]

        return volume

    @property
    def volume(self):
        return self.compute_volume()

    def get_relation(self, rect):
        """
        Specifies the relation betwenn the current hyperrectangle and rect.

        Args:
            rect (Hyperrectangle): the hyperrectangle to compare with

        Returns:
            str: "proper_sub", if rect is a proper subset of self;
                 "proper_super", if rect is a proper superset of self;
                 "equal", if rect is equal to self;
                 "disjoint", if rect and self are disjoint;
                 "neither", if neither of the above is true.
        """

        relation = 'neither'

        # Check if rect is a proper subset of self.
        indicators = [interval_is_subset(rect.bounds[dimension],
                                         self.bounds[dimension])
                      for dimension in rect.dimkeys]

        # If all intervals of rect are subsets of the intervals of self,
        # rect is a subset of self.
        if all(is_subset for (is_subset, _, _) in indicators):
            relation = 'proper_sub'

            # If all intervals of rect are not proper subsets of the intervals
            # of self, rect is equal self.
            if not any(is_proper_sub for (_, is_proper_sub, _) in indicators):
                relation = 'equal'

        # If the intervals for any dimension are disjoint,
        # the hyperrectangles are disjoint as well.
        elif any(are_disjoint for (_, _, are_disjoint) in indicators):
            relation = 'disjoint'

        # If all interval pairs are not disjoint,
        # rect and self are not disjoint either.
        else:
            # Check if self is a proper subset of rect.
            indicators = [interval_is_subset(self.bounds[dimension],
                                             rect.bounds[dimension])
                          for dimension in rect.dimkeys]

            # If all intervals of self are subsets of the intervals of rect,
            # self is a subset of rect.
            if all(is_subset for (is_subset, _, _) in indicators):
                relation = 'proper_super'

        return relation

    def plot(self, axis=None, color='orange'):
        """
        This function will plot the hyperrectangle on the matplotlib axis
        specified. If None is passed in, a new axis will be generated.

        Args:
            axis (matplotlib.axis): The matplotlib axis to plot to
            color (str): The face color of the plotted hyperrectangle
        Returns:
            matplotlib.axis: The axis plotted to
        """
        if self.ndim != 2:
            raise ValueError("Plotting of hyperrectangles only available for"
                             " hyperrectangles of dimension 2")

        if axis is None:
            fig, axis = plt.subplots()

        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)

        key1, key2 = self.dimkeys

        x1, x2 = self.bounds[key1]
        y1, y2 = self.bounds[key2]
        lower_left = x1, y1

        rect = mpatches.Rectangle(lower_left, x2 - x1, y2 - y1,
                                  edgecolor='black', facecolor=color)
        axis.add_patch(rect)
        return axis

    def add_cutouts(self, cutouts=None):
        """
        This function will create a HyperrectangleWithCutouts object from
        this object and the cutout Hyperrectangles passed in. If no
        hyperrectangles are passed in, the cutout set will simply be the
        empty set.

        Args:
            cutouts (list[Hyperrectangle]): The list of hyperrectangles to
                cutout
        Returns:
            HyperrectangleWithCutouts: The rectangle with cutouts
        """
        return HyperrectangleWithCutouts(self, self.dimkeys,
                                         self.name, cutouts)

    def set_dimkeys(self, dimkeys):
        """
        This will change the names of the dimensions of the hyperrectangle.

        Args:
            dimkeys (list): The new names of the dimensions of the
                hyperrectangle.
        """
        new_bounds = {}
        for old_dim, new_dim in zip(self.dimkeys, dimkeys):
            new_bounds[new_dim] = self.bounds[old_dim]
        self.bounds = new_bounds
        MultiDimMixIn.set_dimkeys(self, dimkeys)

    def __hash__(self):
        items = [(dimension, bounds)
                 for dimension, bounds in sorted(self.bounds.items())]
        return hash(tuple(items))

    def __eq__(self, other):
        return sorted(self.bounds.items()) == sorted(other.bounds.items())

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        string = "Hyperrectangle({}):\n".format(self.name)
        for dimension in self.bounds:
            string += "{}: {}\n".format(dimension, self.bounds[dimension])
        return string

    __repr__ = __str__


class Interval(Hyperrectangle):
    """
    A one dimensional hyperrectangle. This will have convenience methods for
    accessing the single dimension associated with the object.
    """
    def __init__(self, a, b, name=None, dimension=0):
        """
        Args:
            a (float): The lower bound of the interval
            b (float): The upper bound of the interval
            name (str): Optionally, the name of the interval
            dimension: Optionally the name of the dimension on which the
                interval is defined, by default this is 0.
        """
        Hyperrectangle.__init__(self, [(a, b)], [dimension], name)

        self.interval = a, b
        self.a, self.b = a, b
        self.dimension = dimension

    def add_cutouts(self, cutouts=None):
        """
        This function will create a IntervalWithCutouts object from
        this object and the cutout Intervals passed in. If no
        intervals are passed in, the cutout set will simply be the
        empty set.

        Args:
            cutouts (list[Interval]): The list of intervals to
                cutout
        Returns:
            IntervalWithCutouts: The interval with cutouts
        """
        return IntervalWithCutouts(self, self.dimension, self.name, cutouts)

    def set_dimension(self, dimension):
        """
        This will set the name of the dimension.

        Args:
            dimension: The name of the dimension on which this is defined
        """
        self.set_dimkeys([dimension])

    def __str__(self):
        string = "Interval({}, ({}, {}))".format(self.name, *self.interval)
        return string

    __repr__ = __str__


class HyperrectangleWithCutouts(MultiDimMixIn):
    """
    This class manages one possible hyperrectangle with cutouts.

    Attributes:
        cutouts (set): a set of hyperrectangles that are cutouts
            of the current object
    """
    def __init__(self, rectangle, dimkeys=None, name=None, cutouts=None):
        """
        Initializes an instance of the HyperrectangleWithCutouts class.

        Args:
            rectangle (Hyperrectangle): The rectangle from which subrectangles
                will be cut out
            dimkeys (list): Optionally, a list of the names of the dimensions
            name (str): Optionally, the name of the hyperrectangle
            cutouts (set): a set of hyperrectangles to cut out
        """
        self.rectangle = rectangle
        self.name = name
        MultiDimMixIn.__init__(self, rectangle.ndim, dimkeys)

        # A set of hyperrectangles, that are proper subsets and need
        # to be cut out of this hyperrectangle.
        if cutouts is not None:
            self.cutouts = cutouts
        else:
            self.cutouts = set()

        for rect in [rectangle] + list(self.cutouts):
            if not self.match(rect):
               raise ValueError("All rectangles must have matching dimension "
                                "names.")

    def _get_interval_divisor(self, dimension):
        """
        Computes the greatest decimal number that divides all possible
        differences between the intervals of the dimension.

        Args:
            dimension (str): the name of the dimension

        Returns:
            float: the greatest interval divisor
        """

        # Write all bounds of the cutouts into one set.
        bounds = set()
        bounds.update(self.rectangle.bounds[dimension])
        for rect in self.cutouts:
            bounds.update(rect.bounds[dimension])

        # Compute the absolute value of all possible differences between the
        # bounds and round them to 4 decimal places in order to avoid binary
        # approximation causing more decimal places. We don't lose any
        # information here since we do not allow the user to define interval
        # bounds with more than 4 decimal palces.
        differences = set()
        for bound1, bound2 in combinations(bounds, 2):
            differences.add(round(abs(bound1 - bound2), 4))

        # If there is only one difference, it must be 1.
        if len(differences) == 1:
            return 1

        # Multiply the differences by 10^4 in order to make them integers.
        int_differences = [int(diff * pow(10, 4)) for diff in differences]

        # Compute the greatest common divisor of the integer differences.
        gcd_diff = gcd(int_differences[0], int_differences[1])
        for i in range(len(int_differences) - 2):
            gcd_diff = gcd(gcd_diff, int_differences[i + 2])
        # Scale the divisor back to the interval [0,1].
        gcd_diff /= pow(10, 4)

        return gcd_diff

    def compute_volume(self):
        """
        Computes the volume of the hyperrectangle. This subtracts the volume
        of any cutout from the superset.

        Returns:
            float: the volume
        """

        volume = self.rectangle.compute_volume()
        for cutout in self.cutouts:
            volume -= cutout.compute_volume()

        return volume

    @property
    def volume(self):
        """
        Computes the volume of the hyperrectangle. This subtracts the volume
        of any cutout.

        Returns:
            float: the volume
        """
        return self.compute_volume()

    def delete_nested_cutouts(self):
        """
        Cleans up the cutout list of hyperrectangle. If there is a subset
        listed, that is already also a subset of an other hyperrectangle in the
        cutout list, it is deleted (i.e. if the cutout list contains
        hyperrectangles like rect1 and rect2 with "rect1 is a subset of rect2",
        we only keep rect2 in order to avoid multiple cutouts of the
        same region.
        """

        new_cutouts = set()

        for i, rect in enumerate(self.cutouts):
            for j, other in enumerate(self.cutouts):
                if i == j:
                    continue
                if rect.get_relation(other) == 'proper_sub':
                    break
            else:
                new_cutouts.add(rect)

        self.cutouts = new_cutouts

    def cover_space_with_hyperrectangles(self):
        """
        Creates a set of Hyperrectangle objects that cover the current
        hyperrectangle without the cutouts.

        Returns:
            set: the set of hyperrectangles
        """

        # If the hyperrectangle doesn't contain any cutouts, the cover is just
        # the hyperrectangle itself.
        if not self.cutouts:
            return {self}

        cover_set = set()
        # Determine the edge length of the hyperrectangles which are used for
        # the cover.
        lengths = [self._get_interval_divisor(dim) for dim in self.dimkeys]

        # This is a list of the subdivisions along each axis (0, 0.1, ..., 1)
        # Goes to 1+length to include the upper bound
        subdivisions = [list(np.arange(0, 1+length, length))
                        for length in lengths]
        # We round to avoid floating point arithmetic errors.
        subdivisions = [[round(x, 4) for x in xs] for xs in subdivisions]
        # This will change the above lists into lists of ordered pairs.
        # e.g., [(0, 0.1), (0.1, 0.2), ..., (0.9, 1)]
        all_bounds = [list(zip(divs, divs[1:])) for divs in subdivisions]

        # Iterating over the cartesian product of all the interval bounds
        # along each axis.
        for bounds in product(*all_bounds):
            rect = Hyperrectangle(bounds, dimkeys=self.dimkeys)
            cover_set.add(rect)

        # Right now, the cover set also covers the hyperrectangles from the
        # cutout set. Therefore, the hyperrectangles in the cover set that are
        # subsets of coverrectangles in the cutout set must be deleted.
        for cutout_rect in self.cutouts:
            cover_set = {cover_rect for cover_rect in cover_set
                         if cutout_rect.get_relation(cover_rect)
                         not in ['equal', 'proper_sub']}

        # Try to merge adjacent hyperrectangles in order to reduce the
        # cardinality of the cover set. Iterate over all dimkeys. In each
        # iteration only hyperrectangles are merged along the axis that belongs
        # to this particular dimension.
        for dimension in self.dimkeys:
            should_restart = True
            # While loop that allows us to restart the inner for loop when ever
            # its iterator has been changed.
            while should_restart:
                should_restart = False
                # Iterate over pairs of cover set hyperrectangles.
                for rect1, rect2 in combinations(cover_set, 2):
                    try:
                        union = rect1.create_union(rect2, dimension)
                        cover_set = cover_set.difference({rect1, rect2})
                        cover_set.add(union)
                        should_restart = True
                        break
                    except ValueError:
                        # If we get a value error, it means we can not merge
                        # along this axis.
                        continue

        return cover_set

    def plot(self, axis=None):
        """
        This plots the current hyperrectangle with cutouts to the axis
        specified.

        Args:
            axis (matplotlib.axis): The axis to plot to, if None passed in
                will generate a new axis
        Returns:
            matplotlib.axis: The axis plotted to
        """
        axis = Hyperrectangle.plot(self, axis)
        for rect in self.cutouts:
            axis = rect.plot(axis, color='white')
        return axis

    def set_dimkeys(self, dimkeys):
        """
        This will set the dimension names for this Hyperrectangle and each of
        the cutout rectangles contained in this set.

        Args:
            dimkeys (list): The names of each of the dimensions
        """
        self.dimkeys = dimkeys
        self.rectangle.set_dimkeys(dimkeys)
        for cutout in self.cutouts:
            cutout.set_dimkeys(dimkeys)

    def __str__(self):
        string = str(self.rectangle) + '\n'
        string += "Cutouts\n"
        for cutout in self.cutouts:
            string += '\t' + str(cutout)
        return string

    def __repr__(self):
        return "HyperrectangleWithCutouts({})".format(self.name)

    def __hash__(self):
        rects = [self.rectangle] + list(self.cutouts)
        first_hash = tuple([hash(rect) for rect in rects])
        return hash(first_hash)


class IntervalWithCutouts(HyperrectangleWithCutouts):
    """
    An object constructed for the convenience of accessing single dimensional
    features directly instead of through a dictionary. This acts as a container
    of a single Interval object which acts as a superset and a collection of
    cutout Intervals which are removed from the superset.

    Attributes:
        interval (Interval): The set from which intervals are cut out
        a (float): The lower limit of the containing interval
        b (float): The upper limit of the containing interval
        cutouts (list[Interval]): The sets which are cutout from the main
            interval
        dimension: The name of the dimension (0 if unset)
        name (str): The name of the Interval
    """
    def __init__(self, interval, dimension=0, name=None, cutouts=None):
        """
        Args:
            interval (Interval): The interval from which subintervals
                will be cut out
            dimension: Optionally, The name of the dimension
            name (str): Optionally, the name of the hyperrectangle
            cutouts (set): a set of intervals to cutout
        """
        HyperrectangleWithCutouts.__init__(self, interval, [dimension], name,
                                           cutouts)
        if not isinstance(interval, Interval):
            raise TypeError("The passed in object is not an Interval")
        if not all(isinstance(inter, Interval) for inter in self.cutouts):
            raise TypeError("All cutouts must be of type Interval")
        self.interval = interval
        self.a, self.b = interval.a, interval.b

    def set_dimension(self, dimension):
        """
        This will set the name of the dimension of the interval.

        Args:
            dimension: The name of the dimension
        """
        self.set_dimkeys([dimension])

    def __str__(self):
        return "IntervalWithCutouts({})".format(self.name)
