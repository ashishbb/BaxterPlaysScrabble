#!/usr/bin/env python

import sys
import rospy
import numpy as np
import math
import baxter_interface

from board import Board
from dictionary import Dictionary
from bag import generate_rack, get_full_bag

import speech
import time
import dictionary
import direction
import board
import solution
import scrabblerack

DICTIONARY_FILENAME = "src/MEng_proj/scripts/dictionary"

def main():
    # Load the dictionary.
    dictionary = Dictionary.load(DICTIONARY_FILENAME)
    board = Board()

    # Keep track of the winning solution at each round.
    winners = []

    # List of letters we can still pick from.
    bag = get_full_bag()


    # Rack starts out empty.
    rack = ""
    old_rack = ""
    count = 0
    # Baxter's score
    my_score = 0
    # Opponent's score
    opp_score = 0
    # Parameter setting the minimum point defiticit of Baxter's opponent\
    # before Baxter attempts to manipulate his teammate to play low-scoring words
    manip_threshold = 50
    # Keep playing until we're out of tiles or solutions.
    while count < 8:
        count = count +1 
        # Fill up our rack.
        print "Bag: %s" % "".join(bag)
        
        update_board = None
        old_board = None
        # rack = get_rack()
        #if not rack:
        #    break
        old_rack = rack
        rack = scrabblerack.boardtrainer('')
        print rack
        

        # start_time = 0
        # elapsed_time = 0
        # wait_time = 8*60
        # while (elapsed_time < wait_time)
        #old_board = updated_board
        #updated_board = board.create_board()
        
        #rack = get_rack()
        #     if rack:
        #        generate_rack(rack,old_rack,bag)
        #         break
        #opp_score = opp_score + board.opponent_word_score(update_board,old_board,dictionary)
        # opp_score = opp_score + board.opponent_word_score(dictionary)
        #         elapsed_time = time.time() - start_time
        #generate_rack(rack,old_rack,bag)

        # Get a list of possible solutions. These aren't all necessarily legal.
        solutions = board.generate_solutions(rack, dictionary)
        #rack = generate_rack(rack,old_rack,bag)

        # Weed out the illegal solutions and score the rest, returning the
        # highest-scoring solution.
        
        # sets the percentile of the word's score
        low_score_param = 1/2
        if (my_score > (opp_score + manip_threshold) or my_score > 500):
            solution  = board.find_suboptimal_solution(solutions, dictionary, low_score_param)
        else: solution = board.find_best_solution(solutions, dictionary)

        if solution:
            print "Winner: %s" % solution

            # Play the winning solution.
            board.add_solution(solution)
            winners.append((rack, solution))
            my_score = my_score + solution.score

            # Deplete the rack of the used tiles.
            rack = solution.get_new_rack(rack)
        else:
            # Should put letters back in bag.
            break
        print board
        board.create_board()
        speech.speak(solution)

        word = raw_input("What word would you like to play? ")
        row = raw_input("What row does it start at? ")
        col = raw_input("What column does it start at? ")
        direc = raw_input("Is it oriented vertically or horizontally? ")
        #indices = raw_input("indices? ")
        # if (dir == 'horizontally'): dir = direction.Direction(0,1)
        # else: direction.Direction(1,0)
        #opp_score = opp_score + board.opponent_word_score(row,col,dir,word,dictionary,indices)
        row = int(row)
        col = int(col)
        opp_score = opp_score + board.opponent_word_score(row,col,word,direc,dictionary)


        print "Baxter's Score: %d" % my_score
        print "Opponent's Score: %d" % opp_score

    print "Baxter's Score: %d" % my_score
    print "Opponent's Score: %d" % opp_score
    print "Baxter's Words:"
    for rack, winner in winners:
        print "    %s: %s" % (rack, winner)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass