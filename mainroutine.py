#!/usr/bin/env python

import sys
import rospy
import numpy as np
import math
#import baxter_interface
import time

from board import Board
from dictionary import Dictionary
from bag import generate_rack, get_full_bag

import cv2
import speech
import dictionary
import direction
import board
import solution
import board_vision
import rack_vision
import ScrabbleGUI
#import testeroni
from Classification import CNN_Model

DICTIONARY_FILENAME = "dictionary"

def main():

    sadasdasdas
    gui = ScrabbleGUI.UserInterface()

    # Load the dictionary.
    dictionary = Dictionary.load(DICTIONARY_FILENAME)
    board = Board()

    # Keep track of the winning solution at each round.
    winners = []

    # List of letters we can still pick from.
    bag = get_full_bag()

    # Rack starts out empty. Keep track of current and last rack state.
    rack = ""
    old_rack = ""
    count = 0

    # Keep track of current and last board state,
    update_board = None
    old_board = Board()

    # Baxter's score
    my_score = 0

    #Create classifier
    classify = CNN_Model()

    # Create video feeds
    # cam = cv2.VideoCapture(1)
    # print cam.isOpened()



    # Keep playing until we're out of tiles or solutions.
    while count < 8:
        count+=1
        # Fill up our rack.
        print "Bag: %s" % "".join(bag)
        old_rack = rack

        # Updates rack with current rack from video feed.
        # cam1 = cv2.VideoCapture(1)
        # print cam1.isOpened()
        cam1 = 1
        print "FUCKLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLRADD"
        rack = rack_vision.get_rack(classify,cam1)
        # cam1.release()
        cv2.destroyAllWindows()

        # Get a list of possible solutions. These aren't all necessarily legal.
        solutions = board.generate_solutions(rack, dictionary)

        solution = board.find_best_solution(solutions, dictionary)

        print solution.direction

        #gui.addword(solution.word,solution.row,solution.col,solution.,suggestion = True)

        if solution:
            print "Winner: %s" % solution

            # Play the winning solution.
            board.create_board()
            print("I suggest you play the word:"+solution.word)
            #speech.speak(solution)
        else:
            pass
            # Should put letters back in bag.
            
        print board
        
        # Wait for "Enter" press, signifying the player has completed his/her turn.
        wait = raw_input("Press enter when finished with move")

        # Get word that was just played on the board by fetching the new board state.
        update_board = Board()
        update_board.set_cells(board_vision.get_board(classify,cam1))
        
        move,letter_placed_on_board = board.get_played_word(update_board,old_board)

        print ("The word:"+ move +"was just played.")

        if (move == solution.word):
            print("Player listened to Baxter")
        else:
            print("Bitch defied Baxter")

        print "Baxter's Score: %d" % my_score

        generate_rack(rack,old_rack,bag)

        for char in letter_placed_on_board:
            rack = rack.replace(char,"")

        old_board = update_board
        print ("count:"+str(count))


    print "Baxter's Score: %d" % my_score
    print "Baxter's Words:"
    for rack, winner in winners:
        print "    %s: %s" % (rack, winner)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass