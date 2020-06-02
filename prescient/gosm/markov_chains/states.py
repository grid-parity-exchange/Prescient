#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
states.py

This module houses an abstraction of what a state is for use in Markov chains.
"""

from .values import Value

class State:
    """
    This is an abstraction of a state for a Markov Chain. This is essentially
    a container for a set of key-value pairs.

    We maintain a distinction between a 'State' and the values you might
    get when 'evaluating' a state. This is because we might be discretizing
    a continuous space and the attributes stored in a State object may be
    representative of a multiple different values, e.g., we may have a state
    for the interval [0.1, 0.9] and the corresponding value may be a random
    sample from the interval.

    Toward this end, we have a dictionary contained in self.values which is
    composed of 'unevaluated' values. These values can be builtin objects,
    functions to be called with no arguments, or descendants of the Value class.
    
    To instantiate a State object, when initializing the object, pass
    attribute-value pairs as keyword arguments.

    Example:
        low_forecast = State(forecast=200, error=10)

        The values passed in may be functions as well.
        Assume we have a function normal_sample which when called returns
        a sample from a standard normal distribution. Then the following
        code will model a State whose value is a random sample.

        error_state = State(error=normal_sample)

        There are also a host of Value objects which have their own 
        unique evaluate method.

        error_state = State(error=ConditionalExpectedValue(spline, [0.3, 0.5]))

        This should be a state which takes the conditional expected
        value on the F^-1(0.3), F^-1(0.5).
    """
    def __init__(self, **kwargs):
        self.values = kwargs

    def evaluate(self):
        """
        This function is what converts an unevaluated state into its
        equivalent evaluated state. This is a dictionary which maps the
        original keys to the evaluated values.

        Returns:
            dict: A dictionary of the form {str -> value}
        """
        evaluated = {}
        for key, value in self.values.items():
            # If a member of Value class, and thus has an evaluate method
            if isinstance(value, Value):
                evaluated[key] = value.evaluate()
            # If a function or a function-like
            elif callable(value):
                evaluated[key] = value()
            # Either a builtin or an unrecognized type, we just return it
            else: 
                evaluated[key] = value
        return evaluated

    def apply_function(self, function_dictionary):
        """
        This function takes a dictionary of name-function mappings and applies
        the function to the value stored in each name of the state.

        Functions should be function of one argument.

        For example, if the values dictionary of this State is of the form
        {'forecast': [0.2,0.8], 'error': [0.3, 0.5]}, and we had functions
        forecast_sample and error_sample which expect an interval as argument,
        we could call this function like so:
            state.apply_function({'forecast': forecast_sample,
                                  'error': error_sample})
        
        Args:
            function_dictionary (dict): A dictionary of {name -> function}
                mappings
        Returns:
            dict: A dictionary of the return values of the functions
                {name -> return value}
        """
        values = {}
        for name in function_dictionary:
            values[name] = function_dictionary[name](self.values[name])
        return values

    def match(self, other, fields=None):
        """
        This function will test if another state matches this state on the
        fields passed in. If fields is not passed in, then it will check if
        this state matches in all of the fields. Note this is different from
        if an empty list is passed in the fields argument in which no fields
        are checked.

        This function is based on equality testing between the fields.

        Args:
            other (State): The state to compare it to
            fields (list[str]): A list of fields to compare the state on
        """
        if fields is None:
            fields = list(self.values.keys())

        for field in fields:
            if self.values[field] != other.values[field]:
                return False
        else:
            return True

    def __str__(self):
        string = "State("
        for key, value in self.values.items():
            string += "\n\t{}={}".format(key, value)
        string += "\n)"
        return string

    def __repr__(self):
        string = "State("
        for key, value in self.values.items():
            string += "{}={},".format(key, value)
        
        # To drop the final extra comma
        string = string[:-1]
        string += ')'
        return string

    def __lt__(self, other):
        """
        We impose a dictionary ordering on the states. This is for ease
        in handling large numbers of states and imposing an ordering on the
        transition matrices

        We expect the other state to have the same value names.
        """
        for key in sorted(self.values):
            try:
                if self.values[key] < other.values[key]:
                    return True
                elif self.values[key] > other.values[key]:
                    return False
            except TypeError:
                # If the value doesn't support comparison
                continue
        else:
            # This case happens if all values are equal
            return False

    def __eq__(self, other):
        """
        Two states are the same if all their keys and values are the same.
        """
        values = [self.values[key] for key in sorted(self.values)]
        other_values = [other.values[key] for key in sorted(other.values)]

        return (sorted(self.values.keys()) == sorted(other.values.keys())
                and values == other_values)

    def __hash__(self):
        """
        We define a hash function so that states can be stored
        in a set or dictionary. The hash function is just the
        hash of the list of the values in the dictionary ordered by
        key.
        """
        return hash(tuple([self.values[key] 
                           for key in sorted(self.values.keys())]))

    def __getattr__(self, attr):
        if attr in self.values:
            return self.values[attr]
        else:
            raise AttributeError("{} is not defined".format(attr))


class WordState(State):
    """
    Fun example of a word as a state for use in generating somewhat coherent
    sentences using a Markov Chain. 

    This has a modified evaluate method to just return the word.

    This will be instantiated with a word.

    Args:
        word (str): A word (duh!)
    """

    def __init__(self, word):
        self.word = word
        State.__init__(self, word=word)

    def evaluate(self):
        return self.word


class StateDescription:
    """
    A general description of what a state is.
    This is internally a dictionary mapping strings which are attributes
    of a state to a method for determining what the state of each attribute is

    We pass the descriptors as keyword arguments,e.g., to construct a
    description, we might write
        description = StateDescription(forecast=compute_forecast_state)

    This class overloads the addition operator, so you can combine state
    descriptions like so

    error_forecast_description = (StateDescription(error=error_state)
                                  + StateDescription(forecast=forecast_state))

    Args:
        kwargs (dict): A dictionary mapping names (or collections of
            names) to functions which when passed in the names will return a
            value. The return of this function is ideally a Value which can
            then be evaluated, but may also be any type.

    Example:
        A StateDescription object can be used to generate states where the
        states are quantile intervals.
        We can have a function to determine which interval something is in.

        bounds = [(0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        def interval(x):
            for a, b in bounds:
                if a <= x <= b:
                    return (a, b)
       
        description = StateDescription(forecast=interval)
    """
    def __init__(self, **kwargs):
        self.config = kwargs

    def to_state(self, class_=State, **kwargs):
        """
        This method will construct a state using the methods
        on passed in keyword arguments. All keyword arguments in the
        state configuration must be passed in. You pass in the type of state
        you want to instantiate as well.

        If we have a StateDescription defined as follows:
            description = StateDescription({('forecast', 'error'):
                error_interval, 'pattern': lambda x: x})
        
        We would call to_state like this:
            description.to_state(forecast=200, error=20, pattern=(1,2,3))
       
        Args:
            class_ (State or subclass of State): The type of state

        Returns:
            State: The state of the configuration
        """
        state_config = {}
        for key in self.config:
            if isinstance(key, tuple):
               values = [kwargs[attr] for attr in key]
               state_config[key] = self.config[key](*values)
            elif isinstance(key, str):
                value = kwargs[key]
                state_config[key] = self.config[key](value)
        return class_(**state_config)

    def keys(self):
        """
        This method returns a list of all the attribute names.

        Returns:
            List[str]: A list of strings corresponding to keys
        """
        names = []
        for name in self.config:
            if isinstance(name, tuple):
                names.extend(name)
            else:
                names.append(name)
        return names

    def __add__(self, other):
        """
        This will combine the descriptions of two StateDescription objects
        meaning the dictionary of two descriptions will be merged.
        If any keys match a new function will be constructed which outputs
        a tuple of the results of each function.

        Returns:
            StateDescription: A merged StateDescription object
        """
        new_config = self.config.copy()
        for key in other.config:
            if key in new_config:
                lfunc, rfunc = self.config[key], other.config[key]
                func = lambda x: lfunc(x), rfunc(x)
                func.__name__ = lfunc.__name__ + '_' + rfunc.__name__
                new_config[key] = func
            else:
                new_config[key] = other.config[key]

        return StateDescription(**new_config)

    def __iadd__(self, other):
        """
        This implements augmented assignment to merge state descriptions.
        """
        self.config = (self + other).config
        return self

    def __str__(self):
        string = "StateDescription("
        for key, func in self.config.items():   
            string += "{}={},".format(key, func.__name__)
        # Remove last trailing comma
        string = string[:-1]
        string += ')'
        return string
