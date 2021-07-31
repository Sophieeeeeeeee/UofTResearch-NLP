"""Calculates the output value from RIPA equation"""
# To use the file, scroll to line 106 and input words you'd like to use

import math


def ripa(x: str, y: str, w: str) -> None:
    """RIPA equation function"""

    lambda_value = 1
    alpha_value = -1
    corpus_size = get_corpus_size()

    c_value = (1 / (math.sqrt(lambda_value))) / \
        math.sqrt((-1 * get_cspmi(corpus_size, x, y) + alpha_value))

    prob_x_w = get_joint_p(corpus_size, x, w)
    prob_y_w = get_joint_p(corpus_size, y, w)

    z_x = get_z_term(x)
    z_y = get_z_term(y)

    print("x word and attribute word joint probability: ", prob_x_w)
    print("y word and attribute word joint probability: ", prob_y_w)
    print("RIPA value: ", c_value * (math.log(prob_x_w / prob_y_w) - z_x + z_y))


def get_z_term(word: str, file='vectors.txt') -> float:
    """Return bias value z of given word"""
    f = open(file, "r")
    for row in f:
        row_lst = row.split()
        if row_lst[0] == word:
            return float(row_lst[-1])


def get_cspmi(corpus_size, x: str, y: str) -> float:
    """Return csPMI(x, y)"""
    return get_pmi(corpus_size, x, y) + math.log(get_joint_p(corpus_size, x, y))


def get_pmi(corpus_size, x: str, y: str) -> float:
    """return PMI(x, y)"""
    return math.log(get_joint_p(corpus_size, x, y) /
                    (get_p(corpus_size, x) * get_p(corpus_size, y)))


def get_joint_p(corpus_size, x: str, y: str, w: int = 10, file: str = 'survey.txt') -> float:
    """Return joint probability of x and y:
       number of times x is followed by y in a window of max w words"""
    joint_occurrence = 0
    f = open(file, "r")
    for row in f:
        x_indices = get_all_index(row, x)
        y_indices = get_all_index(row, y)
        for x_idx in x_indices:
            for y_idx in y_indices:
                if 0 <= y_idx - x_idx <= w:
                    joint_occurrence += 1

    # i am eating lunch because my family is eating lunch, repeated occurrence?
    # x = eating
    # y = lunch
    # x_indices = 2, 8
    # y_indices = 3, 9
    # or number of times they occur in the same doc
    return joint_occurrence / corpus_size


def get_p(corpus_size, word: str, file: str = 'vocab.txt') -> float:
    """Return probability of given word"""
    occurrence = 0
    f = open(file, "r")
    for row in f:
        row_lst = row.split()
        if row_lst[0] == word:
            occurrence = int(row_lst[1])
    return occurrence / corpus_size


def get_all_index(sentence: str, word: str) -> list:
    """Return all indices of this word in this sentence"""
    sentence = sentence.split()
    idx = 0
    lst = []
    for idx in range(0, len(sentence)):
        if sentence[idx] == word:
            lst.append(idx)
    return lst


def get_corpus_size(file: str = 'survey.txt') -> int:
    """Return corpus size (total num of words in file)"""
    size = 0
    f = open(file, "r")
    for row in f:
        size += len(row.split())
    return size


if __name__ == '__main__':
    # x = 'word', y = 'word', w = 'word', change into your own choices
    # run it, prints 3 numbers, first one is p(x, w), second one is p(y, w), last one is RIPA value
    # if outputs math domain error, means joint probability of one word with attribute word is 0,
    # that they have never co-occured, hence log doesn't apply, RIPA can't be calculated
    ripa(x='inperson', y='onlin', w='easili')
