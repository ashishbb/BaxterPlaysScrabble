# Copyright 2011 Lawrence Kesteloot

import random

# http://en.wikipedia.org/wiki/Scrabble_letter_distributions#English
LETTERS = """\
AAAAAAAAAB\
BCCDDDDEEE\
EEEEEEEEEF\
FGGGHHIIII\
IIIIIJKLLL\
LMMNNNNNNO\
OOOOOOOPPQ\
RRRRRRSSSS\
TTTTTTUUUU\
VVWWXYYZ??\
"""

BLANK = "?"

def get_full_bag():
    """Returns a list of letters in the whole bag."""

    return list(LETTERS)

def generate_rack1(rack, bag):
    """Given an existing rack (string) and a bag (list of letters), returns a new
    rack with a full 7 letters. The bag is modified in-place to remove the letters."""

    # Put random letters at the front.
    random.shuffle(bag)

    # Figure out how many letters we need.
    needed_letters = 7 - len(rack)

    # Fill up the rack.
    rack += "".join(bag[:needed_letters])

    # Remove from the bag.
    del bag[:needed_letters]

    print "Rack: %s" % rack

    

    return rack

def generate_rack(new_rack,old_rack,bag):
    remove_from_bag = string_intersection(new_rack,old_rack)
    update_bag(remove_from_bag,bag)

def string_intersection(new_rack, old_rack):
    remove = ""
    for c in new_rack:
        if c not in old_rack:
            remove += c
        old_rack = old_rack.replace(c,"",1)
    return remove
        

def update_bag(s,bag):
    for c in s:
        if c in bag:
            bag.remove(c)


