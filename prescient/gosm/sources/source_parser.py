#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
source_parser.py
"""

import sys

import pandas as pd

import prescient.gosm.parsers.tokenizer as tokenizer
from prescient.gosm.sources import Source, ExtendedSource
from prescient.gosm.sources.segmenter import parse_segmentation_file
from prescient.gosm.sources.upper_bounds import parse_upper_bounds_file
from prescient.gosm.sources.upper_bounds import parse_diurnal_pattern_file
from prescient.gosm.parsers.tokenizer import L_PAREN, R_PAREN, COMMA, SEMICOLON
from prescient.gosm.parsers.tokenizer import EQUALS, ALPHANUM, QUOTED_STRING, ANY_STRING

source_token = tokenizer.TokenFormat("SOURCE", 'Source')
# This token will represent any string of characters without the special
# tokens contained in them
string_token = tokenizer.TokenFormat('string', r'[^(),;=\s"]+')

tokens = [source_token, L_PAREN, R_PAREN, COMMA, SEMICOLON, EQUALS,
          QUOTED_STRING, string_token]


def boolean(string):
    """
    This returns True if the string is "True", False if "False",
    raises an Error otherwise. This is case insensitive.

    Args:
        string (str): A string specifying a boolean
    Returns:
        bool: Whether the string is True or False.
    """
    if string.lower() == 'true':
        return True
    elif string.lower() == 'false':
        return False
    else:
        raise ValueError("Expected string to be True or False")

recognized_keys = {
        'source_type': str,
        'actuals_file': str,
        'forecasts_file': str,
        'segmentation_file': str,
        'capacity_file': str,
        'diurnal_pattern_file': str,
        'is_deterministic': boolean,
        'frac_nondispatch': float,
        'scaling_factor': float,
        'forecasts_as_actuals': boolean,
        'aggregate': boolean,
        'disaggregation_file': str
}

defaults = {
    'segmentation_file': None,
    'capacity_file': None,
    'diurnal_pattern_file': None,
    'is_deterministic': False,
    'frac_nondispatch': 1,
    'scaling_factor': 1,
    'forecasts_as_actuals': False,
    'aggregate': False,
    'disaggregation_file': None
}

required_keys = [
    'source_type',
    'actuals_file',
    'forecasts_file'
]


class ParsedSource:
    """
    This will be just a Python representation of the text specified in a
    source file. To that end, it will simply have a name attribute and a
    dictionary of the key-value mappings

    Attributes:
        name (str): The name of the source
        params (dict): A mapping from attributes of a source to their
            corresponding values
    """
    def __init__(self, name, params):
        self.name = name
        self.params = params
        self._validate()

    def _validate(self):
        """
        This checks if all the keys and parameters are valid.
        It determines if the keys are recognized and if the required keys
        are set raising an error if not done so.
        """
        # We check if it has all the required keys
        for key in required_keys:
            if key not in self.params:
                raise ValueError("Source {} has no {} key".format(
                    self.name, key))
        # We check if all keys are recognized
        for key in self.params:
            if key not in recognized_keys:
                raise ValueError("Source {} has unrecognized key {}".format(
                    self.name, key))

    def to_source(self):
        """
        This will construct an ExtendedSource object from the specification
        of the source in self.params. This will read in the actuals and
        forecasts from the file, segmentation criteria and capacities if these
        are specified. It will also set additional parameters for the source
        if specified as well.

        Returns:
            ExtendedSource: The Source of data
        """
        params = {}
        for key, value in self.params.items():
            try:
                parsed_value = recognized_keys[key](value)
            except ValueError:
                raise ValueError("The key {} for source {} was not set"
                                 " properly.".format(key, self.name))
            params[key] = parsed_value

        for key, value in defaults.items():
            if key not in params:
                params[key] = value

        actual_file = self.params['actuals_file']
        forecast_file = self.params['forecasts_file']
        source_type = self.params['source_type']

        # Parse out actual and forecast file
        if actual_file == forecast_file:
            source = source_from_csv(actual_file, self.name, source_type)
        else:
            # If specified separately, we read them in separately
            act_source = source_from_csv(actual_file, self.name, source_type)
            for_source = source_from_csv(forecast_file, self.name, source_type)
            forecasts = for_source.get_column('forecasts')
            act_source.add_column('forecasts', forecasts)
            source = act_source

        if self.params.get('forecasts_as_actuals'):
            if params.get('is_deterministic', defaults['is_deterministic']):
                source.add_column('actuals', source.get_column('forecasts'))
            else:
                raise ValueError("Only deterministic sources can have their "
                                 "actuals be set to the forecasts.")

        # Parse out segmentation file
        if 'segmentation_file' in self.params:
            seg_file = self.params['segmentation_file']
            criteria = parse_segmentation_file(seg_file)
        else:
            criteria = None

        # Parse out daily capacities
        if 'capacity_file' in self.params:
            cap_file = self.params['capacity_file']
            upper_bounds = parse_upper_bounds_file(cap_file)
        else:
            upper_bounds = None

        # Parse out diurnal pattern for solar sources
        if 'diurnal_pattern_file' in self.params:
            dp_file = self.params['diurnal_pattern_file']
            if source_type != 'solar':
                raise ValueError("The source {} specifies a diurnal pattern "
                                 "file even though it is not a solar source."
                                 .format(self.name))
            diurnal_patterns = parse_diurnal_pattern_file(dp_file)
        else:
            diurnal_patterns = None

        # Parse out disaggregation file
        if params['aggregate']:
            if params['disaggregation_file'] is None:
                raise ValueError("The aggregate option is set to True for "
                                 "source {} but 'disaggregation_file' is not "
                                 "set".format(self.name))
            df = pd.read_csv(params['disaggregation_file'], index_col='source',
                             comment='#')
            disaggregation = dict(df['proportion'])
            params['disaggregation'] = disaggregation

        return ExtendedSource(source, criteria, upper_bounds,
                              diurnal_pattern=diurnal_patterns,
                              source_params=params)


class SourceParser:
    """
    This class will parse out a Sources File to produce equivalent Source
    objects. Source objects are initially just a name with key-value mappings.

    The Sources File is structured according to the following grammar:
        <SourcesFile> ::= {<Source>}
        <Source> ::= Source(<AlphaNum>[,<arguments>]);
        <arguments> ::= <pair> {, <pair>}
        <pair> ::= <AlphaNum>=<QuotedString>
    """
    def __init__(self, tokens):
        self.tokens = tokens
        self.curr_index = 0
        self.curr_token = self.tokens[0]
        # This will hold the already parsed sources
        self.sources = []
        # This will hold the name of the current source being parsed
        self.curr_name = None
        # This will hold the attributes that have been parsed
        self.curr_attrs = {}

    def advance(self):
        self.curr_index += 1
        if self.curr_index == len(self.tokens):
            self.curr_token = 'EOF'
        else:
            self.curr_token = self.tokens[self.curr_index]
            self.curr_line = self.curr_token.line

    def parse(self):
        while True:
            token = self.curr_token
            if token == source_token:
                self.advance()
                self.parse_source()
            elif token == 'EOF':
                break
            else:
                raise ValueError("Expected keyword 'Source' at line {}, got {}"
                    .format(self.curr_line, token.string))
        return self.sources

    def check(self, token, expected_token, error_string):
        """

        """
        if token == 'EOF':
            raise ValueError("Unexpected end of file at line {}".format(
                self.curr_line))
        elif token != expected_token:
            error = "Error: line {} position {}, got '{}'".format(
                token.line, token.position, token.string)
            raise ValueError(error + '\n' + error_string)

    def parse_source(self):
        self.check(self.curr_token, L_PAREN, "Expected '('")
        self.advance()

        name_token = self.curr_token
        self.check(name_token, string_token,
                   "Expected source name (alphanumeric string)")
        self.curr_name = name_token.string

        self.advance()
        next_token = self.curr_token
        if next_token == COMMA:
            self.advance()
            self.parse_arguments()
        self.check(self.curr_token, R_PAREN, "Expected ')'")
        self.advance()
        self.check(self.curr_token, SEMICOLON, "Expected ';'")
        self.advance()

        self.sources.append(ParsedSource(self.curr_name, self.curr_attrs))

        # We need to reset curr_attrs after we have finished a source
        self.curr_attrs = {}

    def parse_arguments(self):
        self.parse_pair()
        while self.curr_token == COMMA:
            self.advance()
            self.parse_pair()

    def parse_pair(self):
        """
        This parses out a <alphanum>=<quote> pair and stores the keyword-value
        in self.curr_attrs
        """
        keyword = self.curr_token
        self.check(keyword, string_token,
                   "Expected keyword (alphanumeric string)")
        self.advance()
        self.check(self.curr_token, EQUALS, "Expected '='")
        self.advance()
        self.check(self.curr_token, QUOTED_STRING, "Expected Quoted Value")
        value = self.curr_token
        self.advance()

        # We drop the quotes from the string as we don't need them after
        # parsing.
        self.curr_attrs[keyword.string] = value.string[1:-1]


def source_from_csv(filename, source_name, source_type):
    """
    Constructs a Source object using the data stored in a csv-file.
    This file should have column headers and be indexed by datetimes stored
    in the first column. It should not have any extraneous tags or lines.

    Args:
        filename (str): The name of the file
        source_name (str): The name of the source
        source_type (str): The type of source
    Returns:
        Source: The source object
    """
    data = pd.read_csv(filename, parse_dates=True, index_col=0,
                       infer_datetime_format=True)

    # We check if the datetimes are formatted properly first
    if not isinstance(data.index, pd.DatetimeIndex):
        raise RuntimeError("Unable to parse out datetimes from file '{}', "
                           "Check that all datetimes are formatted correctly "
                           "and in the first column."
                           .format(filename))

    if data.index.isnull().any():
        print("Warning: datetime column in {} has empty cells, these rows "
              "will be removed.".format(filename), file=sys.stderr)
        # Drop all nonempty datetime rows
        data = data[~data.index.isnull()]


    return Source(source_name, data, source_type)


def sources_from_sources_file(filename):
    """
    This will read and parse out the sources from the sources file as
    described in the documentation. This will construct ExtendedSource objects
    according to the source information and the datetime passed in.

    This will parse out the segmentation criteria and an upper bound.

    A sources file is structured in the following manner:
        name,actuals,forecasts,source_type,segment_file,upper_bounds_file

    Args:
        filename (str): The name of the file
    Returns:
        List[ExtendedSource]: The collection of ExtendedSources
    """
    with open(filename) as f:
        sources = []
        for line in f:
            text, *_ = line.split('#')
            text = text.strip()
            if text:
                name, actual, forecast, s_type, seg, upper = text.split(',')

                if actual == forecast:
                    source = source_from_csv(actual, name, s_type)
                else:
                    actual_source = source_from_csv(actual, name, s_type)
                    forecast_source = source_from_csv(forecast, name, s_type)
                    forecasts = forecast_source.get_column('forecasts')
                    actual_source.add_column('forecasts', forecasts)
                    source = actual_source

                criteria = parse_segmentation_file(seg)
                if upper:
                    upper_bounds = parse_upper_bounds_file(upper)
                else:
                    upper_bounds = None

                sources.append(ExtendedSource(source, criteria, upper_bounds))
    return sources


def sources_from_new_sources_file(filename):
    """
    This will read in and parse out the sources from the newly structured
    sources files designed as of November 28. It will return ExtendedSource
    which has information about the name, forecasts, actuals, source_type,
    segmentation criteria, upper bounds, and other information.

    Args:
        filename (str): The name of the file
    Returns:
        list[ExtendedSource]: A collection of extended sources parsed from
            the file
    """
    file_tokens = tokenizer.tokenize_file(filename, tokens)
    source_parser = SourceParser(file_tokens)
    try:
        parsed_sources = source_parser.parse()
    except ValueError as e:
        error_string = e.args[0]
        raise ValueError("Error in {}".format(filename) + '\n' + error_string)
    return [source.to_source() for source in parsed_sources]


def convert_sources_file(sources_file, new_filename):
    """
    This function will convert an old csv-style source file into the new
    sources file format.

    Args:
        sources_file (str): The name of the file in the old file format
        new_filename (str): The name of the new file to be written
    """
    fields = ['name',
        'actuals_file',
        'forecasts_file',
        'source_type',
        'segmentation_file',
        'capacity_file'
    ]
    with open(sources_file) as reader, open(new_filename, 'w') as writer:
        for line in reader:
            clean_line = tokenizer.gobble_comments(line).strip()
            if clean_line:
                values = clean_line.split(',')
                writer.write('Source({},\n'.format(values[0]))
                equalities = []
                for key, value in zip(fields[1:], values[1:]):
                    if value:
                        equalities.append('\t{}="{}"'.format(key, value))
                writer.write(',\n'.join(equalities))
                writer.write(');\n')

