#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
partitions.py

This module mainly exports a class Partition which will be used to encode
which sources are correlated and therefore should have a copula fit to them.
This will also export a function for parsing a file specifying this partition.
"""

def is_pairwise_disjoint(sets):
    """
    This function will determine if a collection of sets is pairwise disjoint.

    Args:
        sets (List[set]): A collection of sets
    """
    all_objects = set()
    for collection in sets:
        for x in collection:
            if x in all_objects:
                return False
            all_objects.add(x)
    return True


def parse_partition_file(filename):
    """
    This function will parse out a file which lists the partition of sources.
    Two sources are in the same partition if they are expected to be
    correlated.

    The format of this file should be structured as the following example:

        # Partitions File
        # Any lines starting with # are ignored

        <Group 1>:
        -<name_1>
        -<name_2>

        <Group 2>:
        -<name_3>
        -<name_4>

        Singletons:
        -<name_5>
        -<name_6>

    Note that group names are not parsed out and are rather just used for the
    user's own purposes. Also, the Singletons group is not required if there
    are no singletons.

    Args:
        filename (str): The name of the file to parse
    Returns:
        Partition: The partition of the sources
    """


    with open(filename) as f:

        eof = False

        def get_line():
            """
            This function will read in the next line that actually has relevant
            data. This is a line that has something other than a comment and
            is not blank.

            This will eventually cause a StopIteration Error.

            Returns:
                str: The relevant line
            """
            while True:
                line = next(f)
                text, *_ = line.rstrip().split('#')
                text = text.strip()
                if text:
                    return text

        def parse_group():
            """
            This function will parse out a single group from the text file.
            It will advance the file pointer to the line following the last
            element of the group

            Returns:
                List[str]: The list of source names
            """
            group = []
            nonlocal line
            try:
                line = get_line()
                while line.startswith('-'):
                    group.append(line[1:])
                    line = get_line()
            except StopIteration:
                # This means we've reached the end of the file
                nonlocal eof
                eof = True

            return group

        def parse_singletons():
            """
            This function will parse out the special singletons field. It will
            return a list of singleton lists.

            Returns:
                List[List[str]]: The list of singletons.
            """
            singletons = []
            nonlocal line
            try:
                line = get_line()
                while line.startswith('-'):
                    singletons.append([line[1:]])
                    line = get_line()
            except StopIteration:
                # This means we've reached the end of file
                nonlocal eof
                eof = True

            return singletons

        sets = []

        line = get_line()
        while True:
            if eof:
                break

            if line == 'Singletons:':
                sets.extend(parse_singletons())
            else:
                sets.append(parse_group())

        return Partition(sets)

class Partition:
    """
    This class will produce objects which act as a container for disjoint
    sets. This will export methods related to finding the set which contains
    a certain object and other related to the mathematical object that is
    a partition.

    This class overloads __iter__ to iterate through the sets of the partition.
    """
    def __init__(self, sets):
        """
        This will construct a Partition object given a list of iterables.
        It is required that each set be disjoint.

        Args:
            sets (List[iterable]): A list of disjoint collections that can each
                be coerced into a set
        """

        self.sets = [set(collection) for collection in sets]
        if not is_pairwise_disjoint(self.sets):
            raise ValueError("The sets passed in must be pairwise disjoint")

    def find_set(self, x):
        """
        This will find the set which contains the object x, raising an error
        if it is not in any set.

        Args:
            x: The object you wish to find the containing set for
        Returns:
            set: The set containing x
        """
        for collection in self.sets:
            if x in collection:
                return collection
        else:
            raise ValueError("{} is not in any set in the partition".format(x))

    def in_partition(self, x):
        """
        This will check if an object x is in any set in the partition. It
        returns a boolean signifying this fact.

        Args:
            x: The object you wish to check is in the partition
        Returns:
            bool: True if x is in the partition, False otherwise
        """
        return any([x in collection for collection in self.sets])

    def equivalent(self, x, y):
        """
        This checks if x and y are equivalent according to the equivalence
        relation induced by the partition, i.e., they are the same if they
        are in the same set. This raises an error if either are not in the
        partition.

        Args:
            x: The first object in the comparison
            y: The second object in the comparison
        Returns:
            bool: True if they are equivalent
        """
        return self.find_set(x) == self.find_set(y)

    def add_set(self, set_):
        """
        This adds a set to the partition. It first checks that the set is
        disjoint from all the other sets in the partition (raising an error
        if not) and then adds it to the collection.

        Args:
            set_: A collection that can be coerced into a set
        """

        new_set = set(set_)

        # Checking if set_ intersects any set in the partition
        if any(new_set & other for other in self.sets):
            raise ValueError("{} is not disjoint from all other sets"
                             .format(set_))
        self.sets.append(new_set)

    def add_singleton(self, x):
        """
        This adds a singleton set {x} to the list of sets. It first checks if
        x is in any set before adding raising an error if this is true

        Args:
            x: The object that you want to insert
        """
        self.add_set({x})

    def singletons(self):
        """
        This function will return all singletons in the partition. This will
        return the elements themselves, and not the sets of one elements.

        Returns:
            List: The list of singletons
        """
        # We unpack the set into its first element and rest and the set is
        # a singleton if rest has no elements
        return [x for (x, *rest) in self.sets if len(rest) == 0]

    def to_file(self, filename):
        """
        This will write out a partition to a file in the Partitions File
        format.

        Args:
            filename (str): The name of the file to write to
        """
        with open(filename, 'w') as f:
            for set_ in self.sets:

                # Singleton sets have a different format
                if len(set_) == 1:
                    continue

                f.write('Group:\n')
                for element in set_:
                    f.write('-{}\n'.format(element))
                f.write('\n')

            singletons = self.singletons()
            if singletons:
                f.write('Singletons:\n')
                for element in singletons:
                    f.write('-{}\n'.format(element))
                f.write('\n')

    def __iter__(self):
        return iter(self.sets)

    def __repr__(self):
        return '{' + ', '.join(map(repr, self.sets)) + '}'

    def __str__(self):
        string = "{\n"
        for set_ in self.sets:
            string += '\t{\n'
            for element in set_:
                string += '\t\t{},\n'.format(element)
            string += '\t},\n'
        string += '}\n'
        return string
