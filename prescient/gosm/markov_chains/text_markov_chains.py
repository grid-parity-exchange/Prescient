#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import numpy as np

def is_end_word(word):
    """
    Determines if a word is at the end of a sentence.
    """
    if word == 'Mr.' or word == 'Mrs.':
        return False
    punctuation = "!?."
    return word[-1] in punctuation
    

def strip_quotes(word):
    """
    Removes any quotation marks from a string.
    """
    clean_word = word
    if word.startswith('"') or word.startswith('“') or word.startswith('‘'):
        clean_word = clean_word[1:]
    if word.endswith('"') or word.endswith('”') or word.endswith('’'):
        clean_word = clean_word[:-1]
    return clean_word

def increment(mapping, word, other):
    """
    Increments the (word, other) transition in the map.
    Creates entries if they do not exist
    """
    if word in mapping:
        if other in mapping[word]:
            mapping[word][other] += 1
        else:
            mapping[word][other] = 1
    else:
        mapping[word] = {other: 1}

def generate_word_map(filename):
    """
    Generates a transition matrix between words drawn from a file
    where the entry for A[word_i, word_j] is the frequency with which
    word_j follows word_i.
    """
    
    # This dictionary will store for each word a dictionary of other words
    # That the word maps to with the count of each word
    mapping_count_dictionary = {}

    with open(filename) as f:
        last_word = None
        for line in f:
            for word in line.rstrip().split():
                word = strip_quotes(word)
                if not word.strip():
                    continue
                if last_word is None:
                    increment(mapping_count_dictionary, 'START', word)
                    last_word = word
                    continue

                increment(mapping_count_dictionary, last_word, word)
                last_word = word

                if is_end_word(word):
                    increment(mapping_count_dictionary, word, 'END')
                    last_word = None

    mapping_count_dictionary['END'] = {'END':1}
    return mapping_count_dictionary

    
def create_matrix(map_dict):
    """
    Constructs transition matrix from a dictionary with counts for each
    word pair
    """
    n = len(map_dict)
    matrix = np.matrix(np.zeros((n,n)))
    
    for i, word in enumerate(map_dict):
        total_transitions = sum(map_dict[word].values())
        for j, other in enumerate(map_dict):
            matrix[i,j] = map_dict[word].get(other, 0) / total_transitions
        if not(np.isclose(matrix[i,:].sum(),1)):
            print(word, map_dict[word])

    return matrix

def print_sentence(markov_chain):
    markov_chain.reset()
    sentence = ' '.join(list(markov_chain)[1:-1])
    print(sentence)
