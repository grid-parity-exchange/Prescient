#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This module will house a generic tokenizer as the first step in parsing out
the various files used for prescient.
"""

import re

def gobble_comments(line):
    """
    This file will remove any comments in a string.
    This means it will remove any characters which succeed a '#' character
    including the '#' character.

    This will return the part of the string without the comment.

    Args:
        line (str): A string which possibly has comments
    Returns:
        str: The string without comments.
    """
    # Discard any part of the line following a hash
    code, *_ = line.split('#')
    return code


def clear_comments_from_file(file_string):
    """
    This function will remove any comments from a file encoded as a string.
    This will return the file string with all the comments removed.

    Args:
        file_string (str): The string containing an entire file
    Returns:
        str: The string of the file without comments
    """
    lines = []
    for line in file_string.split('\n'):
        cleaned_line = gobble_comments(line)
        lines.append(cleaned_line)
    return '\n'.join(lines)


class TokenFormat:
    """
    This class represents a single token of information.
    To instantiate a token, we will have a name of the token as well as
    a regular expression which describes the format of the token.

    Attributes:
        name (str): The name of the token
        format (str): A regular expression describing the format of the string
    """
    def __init__(self, name, format):
        """

        Args:
            name (str): The name of the token
            format (str): A regular expression describing the format of
                the string
        """
        self.name = name
        self.format = format

    def match(self, string, line, position):
        """
        If the string passed in matches the token format, then this will
        return a Token object containing the part of the string matching the
        token and the index of the string starting the remainder of the string.
        Otherwise, it will return None.

        Args:
            string (str): The string to match on
            line (int): The line the string occurs on
            position (int): The position to start matching from
        Returns:
            tuple: Either a pair containing a Token object and a string with
                the remaining part or None indicating no match.
        """
        match = re.match(self.format, string[position:])
        if match:
            start, end = match.span()
            return (Token(self, string[position:position+end], line, position), 
                    position + end)
        else:
            return None

    def __str__(self):
        return "TokenFormat({},{})".format(self.name, self.format)

    __repr__ = __str__


class Token:
    """
    This class represents an instantiation of a token with a specific format
    string specified by a TokenFormat object and the string which matches the
    token format.

    Attributes:
        name (str): The name of the token
        format (str): A regular expression describing the format of the string
        string (str): The string matching the token format
        line (int): The line number of the line the token occurs on
        position (int): The position in the line at which the token starts
    """
    def __init__(self, token_format, string, line, position):
        """
        Args:
            token_format (TokenFormat): The format spec of the Token
            string (str): The string matching the token format
            line (int): The line number of the line the token occurs on
            position (int): The position in the line at which the token starts
        """
        self.name = token_format.name
        self.format = token_format.format
        self.string = string
        self.line = line
        self.position = position

    def __str__(self):
        return "Token({},'{}')".format(self.name, self.string)

    __repr__ = __str__

    def __eq__(self, other):
        """
        Two objects are the same if they have the same name.
        """
        if isinstance(other, TokenFormat):
            return self.name == other.name
        else:
            return object.__eq__(self, other)


# We define some basic token format objects for use
L_PAREN = TokenFormat('L_PAREN', r'\(')
R_PAREN = TokenFormat('R_PAREN', r'\)')
L_BRACKET = TokenFormat('L_BRACKET', r'\[')
R_BRACKET = TokenFormat('R_BRACKET', r'\]')
COMMA = TokenFormat('COMMA', r',')
SEMICOLON = TokenFormat('SEMICOLON', r';')
EQUALS = TokenFormat('EQUALS', r'=')
INTEGER = TokenFormat('INTEGER', r'[0-9]+')
ALPHANUM = TokenFormat('ALPHANUM', r'[a-zA-Z0-9_]+')
QUOTED_STRING = TokenFormat('QUOTED_STRING', r'".*?"')
ANY_STRING = TokenFormat('ANY_STRING', r'[^\s,]+')


def find_non_whitespace(string, i=0):
    """
    Return index of first non-whitespace character in string starting at index
    i. Returns -1 if there are None.
    """
    for j in range(i, len(string)):
        if not string[j].isspace():
            return j
    else:
        return -1


def tokenize(string, tokens, line):
    """
    This function will break up a string into its corresponding tokens. The
    tokens are specified by the tokens argument.

    This function works by iterating through the tokens and trying to match
    them in order to the start of the string. If it finds a match it breaks
    the string on the match and repeats the process with the remainder of the
    string. If no match is found, an error is raised. Once the end of the
    string is reached, the list of tokens is returned.

    Note that the order of the token formats passed in may change the tokens
    produced.

    Args:
        string (str): The string to tokenize
        tokens (list[TokenFormat]): The list of token formats to check
        line (int): The line number the string occurs on
    Returns:
        list[Token]: The list of Tokens produced from the string
    """
    curr_pos = find_non_whitespace(string)

    all_tokens = []
    # If curr_pos is set to -1, we only have whitespace left in the string.
    while curr_pos != -1:
        # We iteratively break the string into smaller parts by removing
        # tokens from the front of the string.
        for token in tokens:
            match = token.match(string, line, curr_pos)
            if match:
                # If we find a match, we break it off the front, and repeat
                # the process with the remainder of the string.
                next_token, next_pos = match
                next_pos = find_non_whitespace(string, next_pos)
                all_tokens.append(next_token)
                curr_pos = next_pos
                break
        else:
            raise ValueError("No token matches at position {} on line {}"
                .format(curr_pos, line))
    return all_tokens


def tokenize_file(filename, tokens):
    """
    This function will break a file into its corresponding tokens.
    Args:
        filename (str): The name of the file to tokenize
        tokens (list[TokenFormat]): The list of token formats to check
    Returns:
        list[Token]: The list of Tokens produced from the string

    """
    all_tokens = []
    with open(filename) as f:
        for i, line in enumerate(f):
            # This will remove all text after a #
            text, *comments = line.split('#')
            all_tokens.extend(tokenize(line, tokens, i))
    return all_tokens
